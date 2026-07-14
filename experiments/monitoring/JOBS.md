# Remote Experiment Job History

该文档由 `python -m experiments.monitoring.job_registry rebuild` 从 `jobs/*.json`
生成。记录默认永久保留；只有人类明确确认后才能删除。它只服务于实验定位和
检测，不是实验协议或项目方法定义。

| Job | Status | Host | Scheduler / run IDs | Units | Remote root | Description |
|---|---|---|---|---:|---|---|
| `amarel-clutter-best6-10seed-ep150-58171730` | queued | `amarel` | 58171730 | 60 | `/cache/home/js3269/projects/FAW_RNN-clutter-best6-10seed` | Clutter six frozen best models, seeds 1-10, 150 full epochs, no early stopping |
| `amarel-minigrid-redblue-200m-58165756` | running | `amarel` | 58165756 | 2 | `/cache/home/js3269/projects/FAW_RNN-minigrid-accel-200m-20260714` | MiniGrid RedBlueDoors LSTM/GaWF formal 200M accelerated run |
| `amarel-minigrid-memory-200m-58165755` | running | `amarel` | 58165755 | 2 | `/cache/home/js3269/projects/FAW_RNN-minigrid-accel-200m-20260714` | MiniGrid MemoryS7 LSTM/GaWF formal 200M accelerated run |
| `amarel-minigrid-smoke-validator-58165754` | completed | `amarel` | 58165754 | 0 | `/cache/home/js3269/projects/FAW_RNN-minigrid-accel-200m-20260714` | MiniGrid MemoryS7/RedBlueDoors smoke dependency validator |
| `amarel-minigrid-redblue-smoke-58165662` | completed | `amarel` | 58165662 | 2 | `/cache/home/js3269/projects/FAW_RNN-minigrid-accel-200m-20260714` | MiniGrid RedBlueDoors LSTM/GaWF 32k accelerated smoke |
| `amarel-minigrid-memory-smoke-58165661` | completed | `amarel` | 58165661 | 2 | `/cache/home/js3269/projects/FAW_RNN-minigrid-accel-200m-20260714` | MiniGrid MemoryS7 LSTM/GaWF 32k accelerated smoke |
| `amarel-atari-breakout-gru2500k-58159012` | failed | `amarel` | 58159012 | 1 | `/cache/home/js3269/projects/FAW_RNN-multitask-gru-phase0-20260713` | Breakout-only GRU-DRQN 2.5M full-18-action control |
| `amarel-atari-pong-gru2500k-58159011` | failed | `amarel` | 58159011 | 1 | `/cache/home/js3269/projects/FAW_RNN-multitask-gru-phase0-20260713` | Pong-only GRU-DRQN 2.5M full-18-action control |
| `amarel-atari-pong-fs1-stack1-70-58160888` | running | `amarel` | 58160888, 58170421, 58170974, 58170978 | 70 | `/cache/home/js3269/projects/FAW_RNN-multitask-gru-phase0-20260713` | Strict Pong fs1/stack1 full 6-action 1M: 7 models × 2 settings × 5 seeds |
| `amarel-atari-pong-breakout-gru5m-58159010` | completed | `amarel` | 58159010 | 1 | `/cache/home/js3269/projects/FAW_RNN-multitask-gru-phase0-20260713` | Pong + Breakout GRU-DRQN 5M transition-balanced collection |
| `sjc-pong-fscompare1m-c219996` | completed | `sjc-remote` | sjc-pong-fscompare1m-c219996, aim3-pong-fscompare1m-c219996 | 14 | `/home/sjc/FAW_RNN-pong-fscompare1m-c219996` | Pong seed42 1M：7 models × strict fs1/stack1 and fs4/stack4 |
| `sjc-pong-fs1-accelval100k-920e862` | completed | `sjc-remote` | sjc-pong-fs1-accelval-920e862-r3, aim3-pong-fs1-val-920e862-r3 | 7 | `/home/sjc/FAW_RNN-pong-fs1-accelval-9280c36` | Pong strict fs1/stack1 100k acceleration validation, 7 models, seed42 |
| `amarel-pong-fs1-accelval100k-58157960` | cancelled | `amarel` | 58157960 | 7 | `/home/js3269/projects/FAW_RNN-pong-fs1-accelval-9559764` | Cancelled Amarel 7-model Pong fs1/stack1 100k acceleration validation |
| `amarel-pong-fs1-depth2-1m-58145944-58146404` | completed | `amarel` | 58145944, 58146404 | 10 | `/home/js3269/projects/FAW_RNN-pong-fs1-l2-03e8bda` | Pong strict fs1/stack1 depth-2 1M validation, verified 10/10 |

单个 job 的精确日志、结果路径、完成条件和备注位于对应的
`jobs/<id>.json`。
