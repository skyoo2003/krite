# krite-train

Training, export, and study code for the [Krite](https://github.com/skyoo2003/krite) decision model
(torch, Apple MPS or CPU). It builds the training mixtures from pinned Hugging Face datasets, trains
the late-interaction decision tower on mmBERT, exports model directories for `krite serve`, and
applies the pre-registered study and release rules.

```bash
pip install krite-train
python -m krite_train.train --help
python -m krite_train.export --ckpt training/ckpt/late8-broad
python -m krite_train.study --help
```

It uses the same workspace as [krite-bench](https://pypi.org/project/krite-bench/) (`$KRITE_ROOT`,
else the krite checkout, else the current directory): checkpoints go to `training/ckpt/`, study
results to `benchmarks/results/arch/`. Sources, licenses, and rules:
[docs/training-data.md](https://github.com/skyoo2003/krite/blob/main/docs/training-data.md).

License: Apache-2.0.
