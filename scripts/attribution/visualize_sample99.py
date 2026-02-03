#!/usr/bin/env python3
"""
Paper-ready visualization for a single example:
- Single-column friendly width
- Vector PDF output
- Clean typography and minimal decorations
"""

import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

# -----------------------------
# Global style (paper-friendly)
# -----------------------------
mpl.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,   # TrueType fonts in PDF
    "ps.fonttype": 42,
    "font.family": "DejaVu Sans",
})

# Load data
project_root = Path(__file__).parent.parent.parent
with open(project_root / "results/1-order/topk_comparison_Llama-3.2-3B-Instruct.json", "r") as f:
    data = json.load(f)

# Pick sample
sample = next(s for s in data["samples"] if s["sample_idx"] == 99)
tokens = sample["tokens"]
greedy_top2 = sample["greedy_top2"]
method_results = sample["method_results"]

def clean_token(t: str) -> str:
    t = t.replace("Ġ", " ").replace("▁", " ")
    if t == "<|begin_of_text|>":
        return "<BOS>"
    return t

METHODS = [
    ("input_causal", "Input Causal"),
    ("depass", "DePASS"),
    ("shapley", "Token Shapley"),
    ("grad_input", r"Grad$\times$Input"),
    ("grad_sam", "GradSAM"),
    ("attention_rollout", "Attn Rollout"),
]

# Colors: restrained and print-safe
C_TOP1 = "#d55e00"   # orange-red
C_TOP2 = "#e69f00"   # orange
C_OK   = "#1b9e77"   # greenish
C_BAD  = "#b2182b"   # dark red
C_GRAY = "#444444"

# -----------------------------
# Layout: single-column figure
# -----------------------------
# ACL single column is roughly 3.25in–3.4in wide.
fig_w = 3.35
fig_h = 6.2   # tall but still reasonable in a single column
fig = plt.figure(figsize=(fig_w, fig_h))
gs = fig.add_gridspec(
    nrows=2, ncols=1,
    height_ratios=[1.15, 1.0],
    left=0.06, right=0.98, top=0.98, bottom=0.06, hspace=0.25
)

ax_tok = fig.add_subplot(gs[0])
ax_tab = fig.add_subplot(gs[1])
for ax in (ax_tok, ax_tab):
    ax.set_axis_off()

# -----------------------------
# Part 1: token sequence
# -----------------------------
ax_tok.text(0.0, 1.02, "Example: Greedy top-2 on IoI (Sample 99)",
            ha="left", va="bottom", fontweight="bold", transform=ax_tok.transAxes)

# Small legend (compact)
legend_y = 0.94
for j, (lab, col) in enumerate([("Top-1", C_TOP1), ("Top-2", C_TOP2)]):
    x0 = 0.00 + j * 0.22
    ax_tok.add_patch(FancyBboxPatch(
        (x0, legend_y - 0.045), 0.06, 0.05,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor="white", edgecolor=col, linewidth=1.2,
        transform=ax_tok.transAxes
    ))
    ax_tok.text(x0 + 0.075, legend_y - 0.02, lab, ha="left", va="center",
                color=C_GRAY, transform=ax_tok.transAxes)

# Token row splitting: auto by length to avoid ugly overflow
cleaned = [clean_token(t) for t in tokens]

# simple heuristic: split near middle
split_idx = len(tokens) // 2
row1 = list(range(0, split_idx))
row2 = list(range(split_idx, len(tokens)))

def draw_row(ax, idxs, y, font=8):
    # compute relative widths proportional to token length, but capped
    lens = np.array([max(2, min(10, len(cleaned[i].strip()) + 1)) for i in idxs], dtype=float)
    lens = lens / lens.sum()
    gap = 0.006
    total_gap = gap * (len(idxs) - 1)
    widths = (1.0 - total_gap) * lens

    x = 0.0
    h = 0.20
    for w, i in zip(widths, idxs):
        if i == greedy_top2[0]:
            ec, lw = C_TOP1, 1.6
        elif i == greedy_top2[1]:
            ec, lw = C_TOP2, 1.6
        else:
            ec, lw = "#bbbbbb", 0.8

        ax.add_patch(FancyBboxPatch(
            (x, y - h/2), w, h,
            boxstyle="round,pad=0.008,rounding_size=0.02",
            facecolor="white", edgecolor=ec, linewidth=lw,
            transform=ax.transAxes
        ))
        ax.text(x + w/2, y, cleaned[i], ha="center", va="center",
                fontsize=font, color=C_GRAY, transform=ax.transAxes)
        x += w + gap

draw_row(ax_tok, row1, y=0.68, font=8)
draw_row(ax_tok, row2, y=0.40, font=8)

# Prompt line (short and light)
prompt = sample.get("prompt", "")
answer = sample.get("answer", "")
prompt_line = f'Prompt → {answer}'
ax_tok.text(0.0, 0.05, prompt_line, ha="left", va="bottom",
            fontsize=8, color="#666666", transform=ax_tok.transAxes)

# -----------------------------
# Part 2: method comparison table
# -----------------------------
ax_tab.text(0.0, 1.02, "Top-2 selected by each attribution method",
            ha="left", va="bottom", fontweight="bold", transform=ax_tab.transAxes)

# Column anchors
x_method = 0.00
x_t1 = 0.46
x_t2 = 0.78
x_status = 0.94

# Header
y0 = 0.92
ax_tab.plot([0.0, 1.0], [y0, y0], color="#999999", lw=0.8, transform=ax_tab.transAxes)
ax_tab.text(x_method, y0 + 0.03, "Method", ha="left", va="bottom", fontweight="bold", transform=ax_tab.transAxes)
ax_tab.text(x_t1, y0 + 0.03, "Top-1", ha="center", va="bottom", fontweight="bold", transform=ax_tab.transAxes)
ax_tab.text(x_t2, y0 + 0.03, "Top-2", ha="center", va="bottom", fontweight="bold", transform=ax_tab.transAxes)
ax_tab.text(x_status, y0 + 0.03, "Match", ha="center", va="bottom", fontweight="bold", transform=ax_tab.transAxes)

def status_of(res):
    if res["exact_match"]:
        return "EXACT"
    if res["set_match"]:
        return "SET"
    if res["top1_match"]:
        return "TOP1"
    return "NONE"

row_h = 0.12
start_y = 0.84

for r, (key, name) in enumerate(METHODS):
    res = method_results[key]
    top2 = res["top2"]
    y = start_y - r * row_h

    # light row separator
    ax_tab.plot([0.0, 1.0], [y - 0.055, y - 0.055], color="#eeeeee", lw=0.7, transform=ax_tab.transAxes)

    # method name
    ax_tab.text(x_method, y, name, ha="left", va="center", transform=ax_tab.transAxes)

    # top1/top2 text (compact: idx + token)
    t1, t2 = top2[0], top2[1]
    t1_ok = (t1 == greedy_top2[0])
    t2_ok = (t2 == greedy_top2[1])

    ax_tab.text(x_t1, y, f"[{t1}] {cleaned[t1]}", ha="center", va="center",
                fontsize=8, color=(C_OK if t1_ok else C_BAD), transform=ax_tab.transAxes)
    ax_tab.text(x_t2, y, f"[{t2}] {cleaned[t2]}", ha="center", va="center",
                fontsize=8, color=(C_OK if t2_ok else C_BAD), transform=ax_tab.transAxes)

    # status badge (no emojis, print-safe)
    st = status_of(res)
    if st == "EXACT":
        ec = C_OK
    elif st in ("SET", "TOP1"):
        ec = "#666666"
    else:
        ec = C_BAD

    ax_tab.add_patch(FancyBboxPatch(
        (x_status - 0.06, y - 0.03), 0.12, 0.06,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor="white", edgecolor=ec, linewidth=1.0,
        transform=ax_tab.transAxes
    ))
    ax_tab.text(x_status, y, st, ha="center", va="center",
                fontsize=7.5, color=ec, fontweight="bold", transform=ax_tab.transAxes)

# Ground truth footer
gt = f"Greedy: [{greedy_top2[0]}] {cleaned[greedy_top2[0]]} ; [{greedy_top2[1]}] {cleaned[greedy_top2[1]]}"
ax_tab.text(0.0, 0.02, gt, ha="left", va="bottom",
            fontsize=8, color="#444444", transform=ax_tab.transAxes)

# Save: prefer PDF for paper, also export PNG for quick preview
out_dir = project_root / "results/1-order/visualizations"
out_dir.mkdir(parents=True, exist_ok=True)
pdf_path = out_dir / "sample_99_paper.pdf"
png_path = out_dir / "sample_99_paper.png"

fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(png_path, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")