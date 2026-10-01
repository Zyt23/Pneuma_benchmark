#!/usr/bin/env python3
"""Legacy LoRA runner for public Uni2TS Moirai 1.0 and 1.1 base backbones.

Native forward/distribution and packing are used unchanged. Adapter code is
the standard W x + alpha/r * B A x parameterization (B initialized to zero).
Only chronological train examples update adapters; validation main-table loss
selects the checkpoint; test labels are used solely for final scoring.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ANCHORS = ["predict_2_3", "predict_4_5", "predict_5_6", "predict_8_9"]


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--version", choices=["1.0", "1.1"], required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--compact-root", type=Path, required=True)
    p.add_argument("--aligned-index-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--anchors", nargs="+", default=ANCHORS)
    p.add_argument("--seq-len", type=int, default=460)
    p.add_argument("--pred-len", type=int, default=20)
    p.add_argument("--train-per-anchor", type=int, default=512)
    p.add_argument("--val-per-anchor", type=int, default=64)
    p.add_argument("--num-steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=16)
    p.add_argument("--lora-dropout", type=float, default=0)
    p.add_argument("--weight-decay", type=float, default=.01)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=20260816)
    p.add_argument("--max-test-samples", type=int, default=0)
    return p.parse_args()


def emit(value):
    print(json.dumps(value, sort_keys=True), flush=True)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def evenly(ids, maximum):
    return ids if len(ids) <= maximum else ids[np.linspace(0, len(ids)-1, maximum, dtype=np.int64)]


def score(pred, truth):
    diff = np.asarray(pred, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    if not np.isfinite(diff).all():
        raise FloatingPointError("Nonfinite prediction or truth")
    mae = float(np.abs(diff).mean())
    mse = float(np.square(diff).mean())
    rmse = float(np.sqrt(mse))
    return {"mae": mae, "mse": mse, "rmse": rmse, "loss": (mae+rmse)/2}


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha, dropout):
        super().__init__()
        self.base = base
        self.lora_A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + nn.functional.linear(
            nn.functional.linear(self.dropout(x), self.lora_A), self.lora_B
        ) * self.scaling


def inject_lora(module, args):
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    targets = [(name, layer) for name, layer in module.named_modules()
               if isinstance(layer, nn.Linear) and not name.startswith(("param_proj", "out_proj"))]
    for name, layer in targets:
        parent_path, _, child = name.rpartition(".")
        parent = module.get_submodule(parent_path) if parent_path else module
        setattr(parent, child, LoRALinear(layer, args.lora_r, args.lora_alpha, args.lora_dropout))
    return [name for name, _ in targets]


class Forecaster:
    def __init__(self, args):
        self.args = args
        if args.version == "2.0":
            from uni2ts.model.moirai2 import Moirai2Module, Moirai2Forecast
            self.module = Moirai2Module.from_pretrained(str(args.model_path)).cuda()
            self.forecast = Moirai2Forecast(module=self.module, prediction_length=args.pred_len,
                context_length=args.seq_len, target_dim=1, feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0).cuda()
            self.patch_size = self.module.patch_size
        else:
            from uni2ts.model.moirai import MoiraiModule, MoiraiForecast
            self.module = MoiraiModule.from_pretrained(str(args.model_path)).cuda()
            self.forecast = MoiraiForecast(module=self.module, prediction_length=args.pred_len,
                context_length=args.seq_len, patch_size=32, target_dim=1,
                feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0).cuda()
            self.patch_size = 32
        self.module.eval()

    def inputs(self, past, future=None):
        past = past.unsqueeze(-1)
        observed = torch.isfinite(past)
        past = torch.nan_to_num(past)
        pad = torch.zeros(past.shape[:2], dtype=torch.bool, device=past.device)
        kwargs = {}
        if future is not None:
            future = future.unsqueeze(-1)
            kwargs = {"future_target": future,
                      "future_observed_target": torch.isfinite(future)}
        return self.forecast._convert(self.patch_size, past, observed, pad, **kwargs)

    def distribution(self, past, future=None):
        items = self.inputs(past, future)
        distr = self.module(*items, torch.full_like(items[3], self.patch_size))
        return distr, items

    def native_loss(self, past, future):
        if self.args.version != "2.0":
            distr, items = self.distribution(past, future)
            target, observed, _, _, _, prediction = items
            nll = -distr.log_prob(target)
            mask = observed & prediction.unsqueeze(-1)
            return nll[mask].mean()
        items = self.inputs(past)
        raw, _ = self.module(*items, training_mode=True)
        ctx_tokens = self.forecast.context_token_length(self.patch_size)
        pred = raw[:, ctx_tokens-1].reshape(-1, self.module.num_predict_token,
            self.module.num_quantiles, self.patch_size)
        pred = pred.permute(0, 2, 1, 3).flatten(-2)[..., :self.args.pred_len]
        target, observed, sample, _, variate, prediction = items
        loc, scale = self.module.scaler(target, observed & ~prediction.unsqueeze(-1), sample, variate)
        # Univariate packed scaler has one loc/scale repeated per token.
        normalized_future = (future - loc[:, ctx_tokens-1, :1]) / scale[:, ctx_tokens-1, :1]
        err = normalized_future[:, None, :] - pred
        q = torch.tensor(self.module.quantile_levels, device=past.device)[None, :, None]
        return torch.maximum(q*err, (q-1)*err).mean()

    def point(self, past):
        if self.args.version != "2.0":
            distr, _ = self.distribution(past)
            ctx_tokens = self.forecast.context_token_length(self.patch_size)
            # Official probabilistic forecast: sample median. Analytic means of
            # heavy-tailed LogNormal components can explode with tiny weights.
            # Fork RNG makes validation and paired tests exactly reproducible,
            # without altering the RNG used by native training losses.
            with torch.random.fork_rng(devices=[past.device]):
                torch.manual_seed(self.args.seed)
                samples=distr.sample(torch.Size([100]))
            future=samples[:,:,ctx_tokens:,:self.patch_size].flatten(2)[:,:,:self.args.pred_len]
            return future.median(dim=0).values
        output = self.forecast(past.unsqueeze(-1), torch.isfinite(past).unsqueeze(-1),
            torch.zeros_like(past, dtype=torch.bool))
        return output.mean(dim=1).reshape(len(past), self.args.pred_len)

    @torch.no_grad()
    def predict(self, raw):
        self.module.eval()
        past = np.ascontiguousarray(raw[:, :self.args.seq_len].transpose(0, 2, 1)).reshape(-1, self.args.seq_len)
        output=[]
        for start in range(0, len(past), self.args.eval_batch_size):
            batch = torch.from_numpy(past[start:start+self.args.eval_batch_size]).cuda()
            output.append(self.point(batch).float().cpu().numpy())
        return np.concatenate(output).reshape(len(raw), raw.shape[2], self.args.pred_len).transpose(0,2,1)


def load_data(args):
    train, val, test, inventory = [], {}, {}, {}
    for anchor in args.anchors:
        cache = args.compact_root / anchor / args.dataset / "qar_compact_shiftN80.npz"
        with np.load(cache) as data:
            raw = np.asarray(data["x"], dtype=np.float32)
        ixroot = args.aligned_index_root / anchor / args.dataset
        indices = {split: np.load(ixroot / f"{split}_indices.npy") for split in ["train", "val", "test"]}
        assert not set(indices["train"]) & set(indices["val"])
        assert not set(indices["train"]) & set(indices["test"])
        assert not set(indices["val"]) & set(indices["test"])
        chosen = {"train": evenly(indices["train"], args.train_per_anchor),
                  "val": evenly(indices["val"], args.val_per_anchor), "test": indices["test"]}
        if args.max_test_samples:
            chosen["test"] = chosen["test"][:args.max_test_samples]
        arrays = {s: np.ascontiguousarray(raw[ids, :args.seq_len+args.pred_len]) for s,ids in chosen.items()}
        train.append(arrays["train"].transpose(0,2,1).reshape(-1,args.seq_len+args.pred_len))
        val[anchor] = arrays["val"]
        test[anchor] = arrays["test"]
        inventory[anchor] = {"n_features": raw.shape[2], "cache_path": str(cache),
            "counts": {s:len(ids) for s,ids in chosen.items()},
            "indices_sha256": {s:sha256(ixroot/f"{s}_indices.npy") for s in chosen}}
        out = args.output_dir / "indices" / anchor
        out.mkdir(parents=True)
        for split, ids in chosen.items():
            np.save(out / f"{split}.npy", ids)
    return np.concatenate(train), val, test, inventory


def evaluate(forecaster, arrays, stage, save=False):
    rows=[]
    for anchor, raw in arrays.items():
        prediction=forecaster.predict(raw)
        truth=raw[:,forecaster.args.seq_len:forecaster.args.seq_len+forecaster.args.pred_len]
        row={"model":f"Moirai-{forecaster.args.version}-R-base", "dataset":forecaster.args.dataset,
            "anchor":anchor, "stage":stage,"n_samples":len(raw), "n_features":raw.shape[2],
            **score(prediction,truth)}
        rows.append(row)
        if save:
            target=forecaster.args.output_dir/stage/anchor
            target.mkdir(parents=True)
            np.save(target/"pred.npy",prediction.astype(np.float32))
            np.save(target/"true.npy",truth.astype(np.float32))
            save_json(target/"metrics.json",row)
            emit(row)
    return rows, float(np.mean([r["loss"] for r in rows]))


def adapter_state(module):
    return {name:p.detach().cpu().clone() for name,p in module.named_parameters() if p.requires_grad}


def frozen_hash(module):
    digest=hashlib.sha256()
    for name,p in module.named_parameters():
        if not p.requires_grad:
            digest.update(name.encode())
            digest.update(p.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    args=arguments()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    started=time.time()
    train,val,test,inventory=load_data(args)
    model=Forecaster(args)
    check=torch.from_numpy(train[:2,:args.seq_len]).cuda()
    with torch.no_grad():
        before=model.point(check).clone()
    targets=inject_lora(model.module,args)
    if not targets:
        raise ValueError("No Linear targets found")
    with torch.no_grad():
        after=model.point(check).clone()
    initial_diff=float((before-after).abs().max())
    assert initial_diff == 0
    frozen_before=frozen_hash(model.module)
    count=sum(p.numel() for p in model.module.parameters())
    trainable=sum(p.numel() for p in model.module.parameters() if p.requires_grad)
    config={**vars(args),"model":f"Moirai-{args.version}-R-base","inventory":inventory,
        "model_sha256":sha256(args.model_path/"model.safetensors"), "targets":targets,
        "trainable_parameters":trainable,"total_parameters":count,"patch_size":model.patch_size,
        "native_loss":"mixture_negative_log_likelihood" if args.version!="2.0" else "normalized_quantile_pinball",
        "point_estimator":"100_sample_median_fixed_seed" if args.version!="2.0" else "mean_of_9_quantiles_as_main_table",
        "checkpoint_selection":"mean_anchor_validation_(MAE+RMSE)/2_trained_steps100_to1000;step0_separate",
        "excluded_targets":["param_proj", "out_proj", "non_Linear_patch_projections"],
        "native_model_config":json.loads((args.model_path/"config.json").read_text()),
        "lr_schedule":"update s=1..1000: factor=s/100 if s<=100 else (1000-s)/900; first=1e-7,100th=1e-5,last=0",
        "frozen_base_parameters_sha256":frozen_before,
        "initial_lora_max_prediction_difference":initial_diff, "torch_version":torch.__version__,
        "model_size":"base",
        "runner_sha256":sha256(Path(__file__)),
        "preprocessing_status":"legacy_full_window_norm_contains_target_statistics"}
    save_json(args.output_dir/"config.json",config)
    emit({"event":"loaded","train_series":len(train),"trainable":trainable,"total":count})
    zero_rows,zero_loss=evaluate(model,test,"zero_shot",save=True)
    _, baseline_val=evaluate(model,val,"validation")
    history=[{"step":0,"validation_loss":baseline_val}]
    best_val=float("inf")
    best=adapter_state(model.module)
    best_step=0
    optimizer=torch.optim.AdamW([p for p in model.module.parameters() if p.requires_grad],
        lr=args.learning_rate,weight_decay=args.weight_decay)
    def factor(step):
        update=step+1
        if update<=args.warmup_steps:
            return update/max(1,args.warmup_steps)
        return max(0,(args.num_steps-update)/max(1,args.num_steps-args.warmup_steps))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,factor)
    rng=np.random.default_rng(args.seed)
    first_grad=None
    for step in range(1,args.num_steps+1):
        model.module.train()
        batch=torch.from_numpy(train[rng.integers(0,len(train),size=args.batch_size)]).cuda()
        optimizer.zero_grad(set_to_none=True)
        loss=model.native_loss(batch[:,:args.seq_len],batch[:,args.seq_len:])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at {step}: {loss}")
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_([p for p in model.module.parameters() if p.requires_grad],1.)
        if not torch.isfinite(grad):
            raise FloatingPointError(f"Nonfinite gradient at {step}")
        if first_grad is None:
            first_grad=float(grad)
            assert first_grad>0
        optimizer.step()
        scheduler.step()
        if step%25==0:
            emit({"step":step,"native_train_loss":float(loss.detach()),"grad_norm":float(grad),"seconds":time.time()-started})
        if step%args.eval_every==0 or step==args.num_steps:
            _,val_loss=evaluate(model,val,"validation")
            item={"step":step,"validation_loss":val_loss,"native_train_loss":float(loss.detach()),
                "learning_rate":optimizer.param_groups[0]["lr"],"seconds":time.time()-started}
            history.append(item)
            if val_loss<best_val:
                best_val,best_step,best=val_loss,step,adapter_state(model.module)
            item["best_step"]=best_step
            save_json(args.output_dir/"training_history.json",history)
            emit(item)
    assert frozen_hash(model.module)==frozen_before,"Frozen base parameter changed"
    last_adapter_nonzero=any(bool(p.count_nonzero()) for name,p in model.module.named_parameters() if "lora_B" in name)
    assert last_adapter_nonzero
    model.module.load_state_dict(best,strict=False)
    torch.save(best,args.output_dir/"adapter_model.pt")
    tuned_rows,tuned_loss=evaluate(model,test,"lora",save=True)
    summary={"model":config["model"],"dataset":args.dataset,"zero_shot_mean_loss":zero_loss,
        "lora_mean_loss":tuned_loss,"improvement_percent":100*(zero_loss-tuned_loss)/zero_loss,
        "best_step":best_step,"best_validation_loss":best_val,"trainable_parameters":trainable,
        "total_parameters":count,"first_gradient_norm":first_grad,"lora_B_updated":last_adapter_nonzero,
        "all_frozen_parameters_unchanged":True,"frozen_base_parameters_sha256":frozen_before,
        "model_sha256":config["model_sha256"],
        "adapter_sha256":sha256(args.output_dir/"adapter_model.pt"),"rows":zero_rows+tuned_rows,
        "elapsed_seconds":time.time()-started,
        "cuda_peak_allocated_bytes":torch.cuda.max_memory_allocated(),
        "cuda_peak_reserved_bytes":torch.cuda.max_memory_reserved()}
    save_json(args.output_dir/"summary.json",summary)
    with open(args.output_dir/"metrics.csv","w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(summary["rows"][0]))
        writer.writeheader()
        writer.writerows(summary["rows"])
    emit(summary)


if __name__=="__main__":
    main()
