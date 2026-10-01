import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
import timesfm
import os


class Model(nn.Module):
    def __init__(self, configs):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()

        model_path = os.environ.get('TIMESFM_MODEL_PATH', 'google/timesfm-2.5-200m-pytorch')
        # Keep the installed upstream constructor defaults; older code passed
        # the misspelled, unsupported keyword ``torch_complie``.
        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_path)
        self.model.compile(
            timesfm.ForecastConfig(
                max_context=configs.seq_len,
                max_horizon=configs.pred_len,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        outputs = []
        for i in range(x_enc.shape[-1]):
            output, _ = self.model.forecast(
                horizon=self.pred_len,
                inputs=x_enc[:, :, i].cpu().numpy().tolist()
            )
            outputs.append(torch.Tensor(output).to(x_enc.device))
        dec_out = torch.stack(outputs, dim=-1)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'zero_shot_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        return None
