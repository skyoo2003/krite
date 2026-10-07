# krite-bench

Benchmark harness for any engine that serves `POST /v1/systemone`
([Krite Protocol v1](https://github.com/skyoo2003/krite/blob/main/docs/protocol/v1.md)). It measures
quality, calibration, option-order invariance, question interference, latency, and memory under
[docs/benchmark-spec.md](https://github.com/skyoo2003/krite/blob/main/docs/benchmark-spec.md). It
talks to engines only over HTTP and contains no ML code.

```bash
pip install krite-bench
```

## Workspace

Every path resolves against a workspace with the Krite repository layout: `$KRITE_ROOT`, else the
krite checkout the package sits in, else the current directory.

| Path | Content |
|---|---|
| `benchmarks/baselines/engines.toml` | engines to measure (you write it) |
| `benchmarks/data/` | suites built by `krite-bench data` (dataset text stays local) |
| `benchmarks/results/baselines/` | result rows (`--out` to change) |

An engine needs a name and a port; start it yourself before measuring:

```toml
[krite-v1]
port = 8147
model = "krite-0.15b-v1"
backend = "candle-metal"
precision = "fp32"
start = "krite serve --port 8147 --model /path/to/krite-0.15b-v1"
```

## Commands

```bash
krite-bench env                                   # environment record (required before measuring)
krite-bench data [--only agnews-choice] [--verify]
krite-bench quality --engine krite-v1 [--langs en] [--memory]
krite-bench calibrate --engine krite-v1
krite-bench invariance --engine krite-v1 [--n 100]
krite-bench interference --engine krite-v1
krite-bench latency --engines krite-v1 [--mode burst|sustained] [--cells primary|aux|matrix] [--cache cold|warm]
krite-bench report --dest baselines.md
```

`latency --cache startup` and `encoder` start processes through the repository's
`benchmarks/baselines/serve.sh` and `target/release/krite`; they need a checkout.

License: Apache-2.0.
