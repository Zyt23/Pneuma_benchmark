# Supplementary experiment inventory

These runners were recovered from the experiment workspace. They are published
as executable source, not as a guarantee that default flags reproduce a
particular archived result. Use the original run configuration and matching
cache/checksum for numerical reproduction. Only core smoke tasks were executed
during release preparation; supplementary code was checked for syntax/imports
where dependencies were available.

| Experiment | Entry point | Required context |
| --- | --- | --- |
| ROCKET classifiers | `scripts/classification/run_qar_rocket_classifiers.py` | Classification caches; explicit split convention; aeon |
| HIVE-COTE2 / InceptionTime | `scripts/classification/run_qar_aeon_classifiers.py` | Same cache format; model-specific environment; seed |
| CSDI forecasting | `scripts/reviewer_baselines/run_csdi_forecast.py` | Forecast cache and upstream CSDI source |
| MOMENT reconstruction detection | `scripts/reviewer_baselines/run_moment_ad.py` | Classification cache and MOMENT weights; resampling/window settings |
| Statistical anomaly baselines | `scripts/reviewer_baselines/run_statistical_ad.py` | Normal training flights, test raw scores, aggregation settings |
| DCdetector | `scripts/reviewer_baselines/run_qar_dcdetector.py` | External source checkout; configured window length and patch sizes |
| ModernTCN detection | `run.py --task_name anomaly_detection --model ModernTCNAD ...` | QAR anomaly cache and explicit training/window settings |
| Event-window sensitivity | `tools/analysis/build_strict_event_window_caches.py` | Private raw conversion statistics and verified source-to-event mapping |
| Proportional-normal window variant | `tools/analysis/build_proportional_normal_window_caches.py` | Already constructed event-window caches; deterministic nested sampling |
| Moirai 1.0 / 1.1 base LoRA, legacy | `tools/finetune/train_moirai_base_lora.py` | Base weights, history caches and aligned split-index manifest |
| Moirai causal-cache variant | `tools/finetune/train_moirai_base_lora_causal.py` | Separately prepared causal cache and its matching index manifest |

The word `causal` in a runner filename cannot make a legacy cache causal.
Do not supply full-window-normalized legacy caches to it and describe the
result as future-free. Moirai adaptation is not the zero-shot foundation
main table, and 1.0/1.1 base are not the main table's Moirai-2.0 checkpoint.

Event-window W30/W50/W80/W100 needs event identity and flight order relative
to that event. Public pseudo filenames cannot recover private maintenance
event boundaries. A gap heuristic, if explicitly enabled by the builder, is
marked non-strict; it is not ground-truth maintenance metadata.

Cross-aircraft transfer and forecasting/detection-correlation analyses also
require archived checkpoint identities, channel order, test indices and score
arrays. They are indexed in the private handoff rather than presented as
verified new results here. A transfer run freezes the source checkpoint and
does not fit preprocessing or thresholds on target test labels.

There are no generated three-seed tables in this source release. Report sample
SD only from recorded independent runs, preserving per-seed outputs. Scripted
illustrations or manually constructed figures are not empirical evidence.
