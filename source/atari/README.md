# Atari Data Source Scripts

This folder contains Atari/RL environment setup and data-source scripts.

- `check_env.py` verifies Gymnasium/ALE availability, action space, and preprocessed
  observation shape.

Task models live in `utils/atari_task_models.py`; Gymnasium wrappers live in
`utils/atari_envs.py`; shared recurrent math stays under `utils/recurrent_cores/`.
