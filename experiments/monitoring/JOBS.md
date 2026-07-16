# Remote Experiment Job History

该文档由 `python -m experiments.monitoring.job_registry rebuild` 从 `jobs/*.json`
生成。记录默认永久保留；只有人类明确确认后才能删除。它只服务于实验定位和
检测，不是实验协议或项目方法定义。

| Job | Status | Host | Scheduler / run IDs | Units | Remote root | Description |
|---|---|---|---|---:|---|---|
| `sjc-clutter-best6-chan1-seed42-ep150-aim3-clutter-c1-s42-ep150` | running | `sjc-remote` | aim3-clutter-c1-s42-ep150, aim3-clutter-c1-s42-ep150 | 6 | `/G/MIMOlab/Codes/FAW_RNN-clutter-best6-chan1` | Clutter six frozen chan=2 best hyperparameters reused with chan=1, seed=42 only, 150 full epochs, no early stopping; checkpoints reserved for later fg-switch analysis; training/checkpoints only |
| `amarel-clutter-best6-chan1-seed01-ep150-58174675` | cancelled | `amarel` | 58174675 | 6 | `/cache/home/js3269/projects/FAW_RNN-clutter-best6-chan1` | Clutter six frozen chan=2 best hyperparameters reused with chan=1, seed=1 only, 150 full epochs, no early stopping; training/checkpoints only |
| `amarel-clutter-best6-chan1-seed42-ep150-58174672` | cancelled | `amarel` | 58174672 | 6 | `/cache/home/js3269/projects/FAW_RNN-clutter-best6-chan1` | Clutter six frozen chan=2 best hyperparameters reused with chan=1, seed=42 only, 150 full epochs, no early stopping; checkpoints reserved for later fg-switch analysis |
| `amarel-clutter-best6-chan1-10seed-ep150-58174660` | cancelled | `amarel` | 58174660, 58174661, 58174662, 58174663, 58174664, 58174665, 58174666, 58174667, 58174668, 58174669 | 60 | `/cache/home/js3269/projects/FAW_RNN-clutter-best6-chan1` | Clutter six frozen chan=2 best hyperparameters reused with chan=1, seeds 1-10, 150 full epochs, no early stopping |
| `sjc-clutter-best6-chan1-10seed-ep150-aim3-clutter-c1-10seed-ep150` | cancelled | `sjc-remote` | aim3-clutter-c1-10seed-ep150, aim3-clutter-c1-10seed-ep150 | 60 | `/G/MIMOlab/Codes/FAW_RNN-clutter-best6-chan1` | Clutter six frozen chan=2 best hyperparameters reused with chan=1, seeds 1-10, 150 full epochs, no early stopping; each valid checkpoint is then evaluated on the strict jointswitch-balanced test set |
| `amarel-clutter-best6-10seed-ep150-58172415` | running | `amarel` | 58172415, 58172416, 58172417, 58172418, 58172419, 58172420, 58172421, 58172422, 58172423, 58172424 | 60 | `/cache/home/js3269/projects/FAW_RNN-clutter-best6-10seed` | Clutter six frozen best models, seeds 1-10, 150 full epochs, no early stopping |
| `amarel-clutter-best6-10seed-ep150-58171730` | cancelled | `amarel` | 58171730 | 60 | `/cache/home/js3269/projects/FAW_RNN-clutter-best6-10seed` | Clutter six frozen best models, seeds 1-10, 150 full epochs, no early stopping |
| `amarel-minigrid-redblue-200m-58165756` | completed | `amarel` | 58165756 | 2 | `/cache/home/js3269/projects/FAW_RNN-minigrid-accel-200m-20260714` | MiniGrid RedBlueDoors LSTM/GaWF formal 200M accelerated run |
| `amarel-minigrid-memory-200m-58165755` | recovering | `amarel` | 58165755, 58177788 | 2 | `/cache/home/js3269/projects/FAW_RNN-minigrid-accel-200m-20260714` | MiniGrid MemoryS7 LSTM/GaWF formal 200M accelerated run |
| `amarel-minigrid-smoke-validator-58165754` | completed | `amarel` | 58165754 | 0 | `/cache/home/js3269/projects/FAW_RNN-minigrid-accel-200m-20260714` | MiniGrid MemoryS7/RedBlueDoors smoke dependency validator |
| `amarel-minigrid-redblue-smoke-58165662` | completed | `amarel` | 58165662 | 2 | `/cache/home/js3269/projects/FAW_RNN-minigrid-accel-200m-20260714` | MiniGrid RedBlueDoors LSTM/GaWF 32k accelerated smoke |
| `amarel-minigrid-memory-smoke-58165661` | completed | `amarel` | 58165661 | 2 | `/cache/home/js3269/projects/FAW_RNN-minigrid-accel-200m-20260714` | MiniGrid MemoryS7 LSTM/GaWF 32k accelerated smoke |
| `amarel-atari-breakout-gru2500k-58159012` | failed | `amarel` | 58159012 | 1 | `/cache/home/js3269/projects/FAW_RNN-multitask-gru-phase0-20260713` | Breakout-only GRU-DRQN 2.5M full-18-action control |
| `amarel-atari-pong-gru2500k-58159011` | failed | `amarel` | 58159011 | 1 | `/cache/home/js3269/projects/FAW_RNN-multitask-gru-phase0-20260713` | Pong-only GRU-DRQN 2.5M full-18-action control |
| `amarel-atari-pong-fs1-stack1-70-58160888` | recovering | `amarel` | 58160888, 58170421, 58170974, 58170978, 58180330 | 70 | `/cache/home/js3269/projects/FAW_RNN-multitask-gru-phase0-20260713` | Strict Pong fs1/stack1 full 6-action 1M: 7 models × 2 settings × 5 seeds |
| `amarel-atari-pong-breakout-gru5m-58159010` | completed | `amarel` | 58159010 | 1 | `/cache/home/js3269/projects/FAW_RNN-multitask-gru-phase0-20260713` | Pong + Breakout GRU-DRQN 5M transition-balanced collection |
| `sjc-pong-fscompare1m-c219996` | completed | `sjc-remote` | sjc-pong-fscompare1m-c219996, aim3-pong-fscompare1m-c219996 | 14 | `/home/sjc/FAW_RNN-pong-fscompare1m-c219996` | Pong seed42 1M：7 models × strict fs1/stack1 and fs4/stack4 |
| `sjc-pong-fs1-accelval100k-920e862` | completed | `sjc-remote` | sjc-pong-fs1-accelval-920e862-r3, aim3-pong-fs1-val-920e862-r3 | 7 | `/home/sjc/FAW_RNN-pong-fs1-accelval-9280c36` | Pong strict fs1/stack1 100k acceleration validation, 7 models, seed42 |
| `amarel-pong-fs1-accelval100k-58157960` | cancelled | `amarel` | 58157960 | 7 | `/home/js3269/projects/FAW_RNN-pong-fs1-accelval-9559764` | Cancelled Amarel 7-model Pong fs1/stack1 100k acceleration validation |
| `amarel-pong-fs1-depth2-1m-58145944-58146404` | completed | `amarel` | 58145944, 58146404 | 10 | `/home/js3269/projects/FAW_RNN-pong-fs1-l2-03e8bda` | Pong strict fs1/stack1 depth-2 1M validation, verified 10/10 |

单个 job 的精确日志、结果路径、完成条件和备注位于对应的
`jobs/<id>.json`。
