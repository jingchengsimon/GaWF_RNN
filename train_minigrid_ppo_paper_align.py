"""Compatibility entry point for the isolated paper-alignment trainer.

The implementation lives in :mod:`train_minigrid_ppo_paper`; this alias is
kept so older launch notes do not accidentally invoke the accelerated pilot.
"""

from train_minigrid_ppo_paper import main


if __name__ == "__main__":
    main()
