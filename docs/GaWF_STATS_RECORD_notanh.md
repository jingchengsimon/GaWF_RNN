# GaWF no-tanh 统计记录

本文件由 `gawf_notanh_refresh.py` 从 `gawf_legacy_notanh` 十个 seed 的独立结构化结果
生成。它是 `docs/GaWF_STATS_RECORD.md` 的 **notanh delta 版本**，不复制旧 GaWF
数值。未列出的 dataset、sampling、reset exclusion 与 seed-level aggregation
口径沿用原记录。

## 模型定义

`gawf_legacy_notanh` 保留 legacy in-loop `LayerNorm → ReLU → Dropout` recurrence，
但将内部 `tanh(preactivation)` 替换为 identity。训练协议要求 `actual_epochs=150`、
`core_rnn_activation=identity`、`core_output_wrap=ln_relu_dropout`、
`gawf_core_semantics=legacy`。

## 结构化事实源

| Path | SHA-256 |
|---|---|
| `/Users/jingchengshi/Desktop/MIMO-Rutgers/1-Codes/aim3_gawf_rnn/results/data/analysis/gawf_legacy_notanh_refresh_10seed_v1/final/fig3/fig3_gate_half_mass_notanh.json` | `723081e37d7d911cce5135810cb8f09c49b4ae5d5beae7f257ef62ce9463d14f` |
| `/Users/jingchengshi/Desktop/MIMO-Rutgers/1-Codes/aim3_gawf_rnn/results/data/analysis/gawf_legacy_notanh_refresh_10seed_v1/final/fig3/gate_and_weight_distributions_1x4_10seed_notanh.json` | `660a77f3f6bec84084429812333626e3786d561b0b734351706ed5f30320745e` |
| `/Users/jingchengshi/Desktop/MIMO-Rutgers/1-Codes/aim3_gawf_rnn/results/data/analysis/gawf_legacy_notanh_refresh_10seed_v1/final/fig6_overall_data/fig6_overall_sector_input_gate_1x3_10seed.json` | `efa551e54e1cf5c5793e9eba72eb30db4260d4561d5c4f17dde0af9a3f89ff39` |
| `/Users/jingchengshi/Desktop/MIMO-Rutgers/1-Codes/aim3_gawf_rnn/results/data/analysis/gawf_legacy_notanh_refresh_10seed_v1/final/supple2/Supple2_input_gate_sign_vs_mag_9sector_10seed_stats.json` | `ba06adf3355eec154f3a16e919fcb406fdd9b4adb4cc33005418ecd82eed8115` |
| `/Users/jingchengshi/Desktop/MIMO-Rutgers/1-Codes/aim3_gawf_rnn/results/data/analysis/gawf_legacy_notanh_refresh_10seed_v1/final/fig7/supple3_seed_level_sign_magnitude_stats.json` | `cb133d21f504f003deeeadfe3b89cd4d024d3e800c869432e1a5ae9204478da4` |
| `/Users/jingchengshi/Desktop/MIMO-Rutgers/1-Codes/aim3_gawf_rnn/results/data/analysis/gawf_legacy_notanh_refresh_10seed_v1/final/fig7/fig7_seed_level_summary.npz` | `4e349f1c46ff7b9c86320e59ad04bc3bb5a7fc0089f3d84e637f384e4095d0b4` |
| `/Users/jingchengshi/Desktop/MIMO-Rutgers/1-Codes/aim3_gawf_rnn/results/data/analysis/gawf_legacy_notanh_refresh_10seed_v1/final/current_records/Supple4_recurrent_current_unit_caption_stats.json` | `733ca03177869fd2b10a7c3072f3294139090b46e7b3acdf441153a61d4566ed` |
| `/Users/jingchengshi/Desktop/MIMO-Rutgers/1-Codes/aim3_gawf_rnn/results/data/analysis/gawf_legacy_notanh_refresh_10seed_v1/final/current_records/Fig8_recurrent_current_connection_caption_stats.json` | `748a140a9e0a0fc133f2a2de88236f2b24edae73b050e37a43ac8b5135be5302` |

## Manuscript statistics added after the initial refresh

All values below use the standard test movie, 32-frame rollouts, reset-excluded frames,
and ten training seeds; uncertainty is seed-level SEM.

- Recurrent-gate endpoints: `g_rec < 0.1` =
  **26.2795% ± 0.3882%**;
  `g_rec >= 0.9` =
  **64.7884% ± 0.7423%**.
- Digit `T->T` recurrent modulation on the shared-|W| support: `W>0` =
  **-0.00989 ± 0.00439**; `W<0` =
  **-0.11759 ± 0.01059**; sign gap (`W>0 - W<0`) =
  **+0.10770 ± 0.00690**.

## 已生成 GaWF-only figures

- `/Users/jingchengshi/Desktop/MIMO-Rutgers/1-Codes/aim3_gawf_rnn/results/save/Figures_notanh/gate_and_weight_distributions_1x4_10seed.pdf`
- `/Users/jingchengshi/Desktop/MIMO-Rutgers/1-Codes/aim3_gawf_rnn/results/save/Figures_notanh/input_gate_sign_vs_mag_sector_delta_zoom_10seed_9sector.pdf`
- `/Users/jingchengshi/Desktop/MIMO-Rutgers/1-Codes/aim3_gawf_rnn/results/save/Figures_notanh/overall_sector_input_gate_1x3_10seed.pdf`
- `/Users/jingchengshi/Desktop/MIMO-Rutgers/1-Codes/aim3_gawf_rnn/results/save/Figures_notanh/rec_gate_disinhibit_and_current_2x3_10seed.pdf`
- `/Users/jingchengshi/Desktop/MIMO-Rutgers/1-Codes/aim3_gawf_rnn/results/save/Figures_notanh/rec_gate_sign_vs_mag_digit_sector_delta_2x4_10seed.pdf`
- `/Users/jingchengshi/Desktop/MIMO-Rutgers/1-Codes/aim3_gawf_rnn/results/save/Figures_notanh/recurrent_current_unit.pdf`

## Deferred until baseline no-wrap completion

- six-model behavior and target-switch figures
- six-model activation ANOVA figure
- GaWF/LSTM/GRU unit-gate comparison row
- cross-scale and H128 comparison figures

## Static; no checkpoint-dependent refresh

- GaWF/CM-MNIST schematic
- dataset-generation protocol
