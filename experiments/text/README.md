# Text experiments

IMDB 与 SentiHood 共用 `utils/text_task_models.py` 和 `utils/recurrent_cores/`。本目录保存
IMDB grid、parameter matching 与 Atari 之外的 text task definitions；Slurm wrappers 位于
`experiments/amarel/`。

GaWF 使用公开模型名 `gawf` 或按 feedback 语义区分的 `gawf_logits`，深度统一由
`--num_layers` 指定。新实验不得发出 `gawf_multi`、`--gawf_layers` 或
`--gawf_multi_feedback_lr_scale`。旧 checkpoint 名只在 loader 中保留兼容。
