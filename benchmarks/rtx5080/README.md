# RTX 5080 attention experiment

Measured on 24 November 2026 in WSL2 (Ubuntu 26.04) on one GeForce RTX 5080 with 16 GB VRAM. The configured GPU power limit was 280 W. The benchmark allocated a 65,536-token context and reported 15.26 GB of GPU memory across all three variants.

| Workload | Eight blocks before | Four blocks for sliding attention | Eight blocks after |
| --- | ---: | ---: | ---: |
| Hash-map explanation, decode tok/s | 180.13 | 180.67 | 180.11 |
| Python LRU cache, decode tok/s | 165.78 | 169.06 | 165.63 |
| 20K-token prompt, prefill tok/s | 4,161.56 | 4,158.48 | 4,169.14 |

Decode values are medians of six 512-token generations per workload and variant. Each prefill value comes from one run. The order was control, candidate, control. The candidate changed the CUDA cluster size for the 1,024-token sliding window from eight blocks to four. Full attention kept the same path. The candidate passed the FP64 attention oracle across ring wrap, stale speculative slots and multiple verification widths. The code workload accepted 3.00 tokens per verification on the last candidate run, versus 2.76 on both last control runs.

The results cover this kernel experiment on one machine. The decode prompts were short, so use the 64K setting as the allocated capacity. These readings do not establish full-window decode speed, throughput under concurrent requests, or a speedup over stock ingine.

## Reproduce the eight-block fixture

Set up the RTX 5080 profile as shown in the [README](../../README.md). Stop other compute work on the same GPU, then build and run the benchmark.

```bash
.tools/venv/bin/cmake --build native/build-5080 --target ingine-bench

model=models/Inkling-2-Mini-nvfp4
native/build-5080/apps/bench/ingine-bench \
  --model-dir "$model" \
  --ablation "$model/inkling-2-mini-abliterated.safetensors" \
  --dense-fp8 --kv-fp8 --embed-host --tokens 512 --runs 6 --max-ctx 65536 \
  --prefill-chunk 2048 --expert-plan config/rtx5080-experts.json \
  --prefill-stage-mib 256 --spec --spec-threshold 0.8 --long-prompt 20000
```

[measurements.json](measurements.json) holds all 36 decode measurements, their medians, the three prefill values and the last speculative counts. Regenerate the charts with this command.

```bash
uv run --no-project --with matplotlib==3.10.8 python benchmarks/rtx5080/plot.py
```

## Build provenance

The experiment used the locally installed SM120a control binary. Its SHA-256 is in the measurements. The release's native source tree matches the RTX 5080 branch at `8da4af84e51a2be620ffad3d2871d6214b1e3c61`, based on [ingine `556acc7`](https://github.com/satellitedown/inkling-2-mini-abliterated-ingine/tree/556acc7cf42749536684854a6df820e41258f929). A later clean Release build of that source compiled `ingine-serve` for `sm_120a` with software clusters enabled and passed `--help` with CUDA devices disabled. Its SHA-256 was `0f6298a3cef56f1c860b56cfed459cbeda968e861ebaf7ebbdb7e303bd18a8b3`. The charts show the earlier benchmark binary.
