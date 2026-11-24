"""Reproduce the RTX 5080 kernel comparison charts from the included measurements."""
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

directory = Path(__file__).resolve().parent
measurements = json.loads((directory / "measurements.json").read_text())
runs = measurements["runs"]
colours = ["#547ba6", "#dc779a", "#547ba6"]

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "figure.facecolor": "#141820",
    "axes.facecolor": "#141820",
    "axes.labelcolor": "#d9e5f1",
    "text.color": "#edf5fc",
    "xtick.color": "#d9e5f1",
    "ytick.color": "#b5c5d7",
    "axes.spines.left": False,
    "axes.spines.right": False,
    "axes.spines.bottom": False,
    "axes.spines.top": False,
    "axes.axisbelow": True,
    "grid.color": "#354251",
    "grid.linewidth": 0.6,
})


def chart(title, subtitle, ylabel, ceiling):
    figure, axis = plt.subplots(figsize=(8.6, 4.0), dpi=160)
    figure.subplots_adjust(left=0.15, right=0.97, top=0.76, bottom=0.16)
    figure.text(0.15, 0.91, title, fontsize=20, weight="semibold")
    figure.text(0.15, 0.835, subtitle, fontsize=10.5, color="#b5c5d7")
    axis.set_ylim(0, ceiling)
    axis.set_ylabel(ylabel, fontsize=10)
    axis.grid(axis="y")
    axis.tick_params(length=0, pad=7)
    return figure, axis


figure, axis = chart("Decode on one RTX 5080", "Median · six 512-token runs per variant", "Output tokens/s", 205)
for group, workload in enumerate(("chat", "code")):
    for i, run in enumerate(runs):
        readings = run[workload]["runs"]
        rate = run[workload]["decode_tok_s"]
        x = group + (i - 1) * 0.23
        axis.bar(x, rate, width=0.19, color=colours[i])
        axis.scatter([x + (j - 2.5) * 0.022 for j in range(len(readings))], readings,
                     s=12, color="#f0f7ff", zorder=5, edgecolors="#141820", linewidths=0.3)
        axis.text(x, rate + 4, f"{rate:.1f}", va="bottom", ha="center", fontsize=10, weight="semibold")
axis.set_xticks([0, 1], ["Hash-map explanation", "Python LRU cache"])
axis.set_yticks([0, 50, 100, 150, 200])
figure.legend([mpl.patches.Patch(color=colour) for colour in colours[:2]],
              ["Eight blocks · before and after", "Four blocks · sliding attention only"],
              loc="lower center", ncol=2, frameon=False, fontsize=9)
figure.subplots_adjust(bottom=0.21)
figure.savefig(directory / "decode.png")
plt.close(figure)

figure, axis = chart("20K prompt processing", "One prompt per variant · 65,536-token allocation", "Input tokens/s", 4750)
for i, run in enumerate(runs):
    rate = run["prefill"]["tok_s"]
    axis.bar(i, rate, width=0.48, color=colours[i])
    axis.text(i, rate + 105, f"{rate:,.0f}", va="bottom", ha="center", fontsize=11, weight="semibold")
axis.set_xticks([0, 1, 2], ["Eight blocks\nbefore", "Four blocks\nsliding only", "Eight blocks\nafter"])
axis.set_yticks([0, 1000, 2000, 3000, 4000])
figure.savefig(directory / "prefill.png")
plt.close(figure)
