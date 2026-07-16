# Moving MNIST Clutter

Generates a collection of movie featurs MNIST numbers moving around the screen. One is moving in a straight line, while the move randomly.

### Setting up the environment

Portable Mac and Linux/CUDA definitions live in [`environments/`](environments/README.md).
Use Python 3.11 for the canonical `aim3_rnn` environment; Python 3.14 cannot run
`torch.compile`. Verify a new environment before using it for experiments:

```bash
python verify_aim3_environment.py --profile macos
python verify_aim3_environment.py --profile linux-cuda --compile-smoke
```

You may encounter an error with the `pillow` package that is accessed by `torchvision`. If so, take the following steps:
1. Activate the environment.
2. Uninstall `pillow`
    > pip uninstall pillow
3. Reinstall `pillow`
    > pip install pillow

You may encounter an error with the `matplotlib` package. If so, take the following steps:
1. Activate the environment.
2. Uninstall `matplotlib`
    > pip uninstall matplotlib
3. Reinstall `matplotlib`
    > pip install matplotlib
