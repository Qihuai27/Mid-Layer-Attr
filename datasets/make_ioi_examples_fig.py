#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final IOI appendix example figure (two-column safe, no overlap, text fits prompt box).

Key design:
1) Card height is computed to fit content (in inches).
2) Prompt background box is fixed-width, and text is wrapped + line-spaced to fit the box.
3) No tight_layout(); use explicit geometry.
"""

from __future__ import annotations
import argparse, json, textwrap, math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

# Academic-style color scheme (IOI theme - Green)
COLORS = {
    'ioi': '#27ae60',                # Green
    'background': '#f8f9fa',         # Light gray background
    'card': '#ffffff',               # White card
    'text_primary': '#2c3e50',       # Dark blue-gray
    'text_secondary': '#5a6c7d',     # Medium gray
    'border': '#dce1e6',             # Light border
    'prompt_bg': '#ecf0f1',          # Very light gray for prompt
    'answer_accent': '#e74c3c'       # Red accent for answer
}

def load_samples(path: Path):
    if path.suffix.lower() == ".jsonl":
        out = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out
    elif path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"Unsupported file: {path}")

def wrap_text(s: str, width: int) -> str:
    return "\n".join(textwrap.wrap(
        s, width=width, break_long_words=False, replace_whitespace=False
    ))

def _axes_bbox_from_display(ax, bb):
    inv = ax.transAxes.inverted()
    (x0, y0) = inv.transform((bb.x0, bb.y0))
    (x1, y1) = inv.transform((bb.x1, bb.y1))
    return x0, y0, x1, y1

def _measure_text_axes(fig, ax, text: str, fontsize: float, family: str, linespacing: float):
    t = ax.text(-10, -10, text, fontsize=fontsize, family=family,
                linespacing=linespacing, va="top", ha="left", alpha=0.0)
    fig.canvas.draw()
    bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
    t.remove()
    x0, y0, x1, y1 = _axes_bbox_from_display(ax, bb)
    return (x1 - x0), (y1 - y0)  # axes fraction

def add_prompt_in_box(
    fig, ax,
    box_x0, box_y0, box_w, box_h,    # axes coords
    prompt: str,
    *,
    fontsize: float,
    family: str = "monospace",
    color: str = "#2c3e50",
    bg_color: str = "#ecf0f1",
    bg_alpha: float = 0.65,
    pad_x: float = 0.010,
    pad_y: float = 0.010,
    linespacing_max: float = 1.16,
    linespacing_min: float = 0.98,
    fontsize_min: float | None = None,
    clip_path=None,
    z_bg: int = 3,
    z_text: int = 4,
):
    """
    Fixed background box; text is wrapped to fit width and line-spaced (and if needed resized)
    to fit height. This makes the text feel "compact" and always aligned with the background.
    """
    if fontsize_min is None:
        fontsize_min = max(6.0, fontsize * 0.85)

    # Background
    bg = Rectangle((box_x0, box_y0), box_w, box_h,
                   facecolor=bg_color, alpha=bg_alpha, linewidth=0, zorder=z_bg)
    ax.add_patch(bg)
    if clip_path is not None:
        bg.set_clip_path(clip_path)

    # Estimate wrap width by average character width under current font
    probe = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    fig.canvas.draw()
    probe_w, _ = _measure_text_axes(fig, ax, probe, fontsize, family, linespacing_max)
    avg_char_w = probe_w / max(len(probe), 1)

    inner_w = max(1e-6, box_w - 2 * pad_x)
    wrap_width = max(12, int(inner_w / max(avg_char_w, 1e-6)))

    wrapped = wrap_text(prompt, wrap_width)
    txt = f"\"{wrapped}\""

    inner_h = max(1e-6, box_h - 2 * pad_y)

    # Fit linespacing by binary search (largest that still fits, to avoid sparse look)
    fs = fontsize
    lo, hi = linespacing_min, linespacing_max
    best_ls = lo
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        _, h_mid = _measure_text_axes(fig, ax, txt, fs, family, mid)
        if h_mid <= inner_h:
            best_ls = mid
            lo = mid
        else:
            hi = mid

    # If still overflow, scale fontsize down and refit linespacing
    _, h_best = _measure_text_axes(fig, ax, txt, fs, family, best_ls)
    if h_best > inner_h:
        scale = inner_h / max(h_best, 1e-6)
        fs = max(fontsize_min, fs * scale)

        lo, hi = linespacing_min, linespacing_max
        best_ls = lo
        for _ in range(14):
            mid = 0.5 * (lo + hi)
            _, h_mid = _measure_text_axes(fig, ax, txt, fs, family, mid)
            if h_mid <= inner_h:
                best_ls = mid
                lo = mid
            else:
                hi = mid

    # Draw final text inside box (top-left inner corner)
    x_text = box_x0 + pad_x
    y_text = box_y0 + box_h - pad_y
    t = ax.text(x_text, y_text, txt,
                fontsize=fs, family=family, linespacing=best_ls,
                va="top", ha="left", color=color, zorder=z_text, clip_on=True)
    if clip_path is not None:
        t.set_clip_path(clip_path)

    return t, best_ls, fs, wrap_width

def draw_cards(
    samples,
    outpath: Path,
    n: int = 3,
    wrap_width_hint: int = 40,
    fig_w: float = 3.35,   # two-column: ~\columnwidth inches
    dpi: int = 300,
):
    samples = samples[:n]

    # Typography (two-column tuned)
    fs_title = 12.5
    fs_sub   = 9.2
    fs_body  = 9.2
    fs_ans   = 10.5
    mono_family = "monospace"
    sans_family = "sans-serif"

    # Geometry (inches)
    top_m_in = 0.16
    bot_m_in = 0.14
    gap_in   = 0.18

    # Card paddings
    card_pad_top_in = 0.20
    card_pad_bot_in = 0.16

    # Inner vertical gaps
    gap_title_sub_in = 0.10
    gap_sub_prompt_in = 0.10
    gap_prompt_answer_in = 0.12
    gap_label_value_in = 0.06

    # Fixed prompt box height (inches): choose compact, consistent across cards
    # If你希望更紧凑：降到 0.62~0.66；更宽松：0.72+
    prompt_box_h_in = 0.68

    # Horizontal layout (inches)
    outer_lr_in = 0.18
    accent_pad_in = 0.06
    accent_w_in = 0.04
    content_left_in = 0.18
    content_right_in = 0.10

    # Temp fig for measuring line heights
    tmp_fig = plt.figure(figsize=(fig_w, 1.8), dpi=dpi)
    tmp_ax = tmp_fig.add_axes([0, 0, 1, 1])
    tmp_ax.set_axis_off()

    def line_h_in(fontsize, family):
        w, h = _measure_text_axes(tmp_fig, tmp_ax, "Ag", fontsize, family, 1.0)
        return h * 1.8  # axes->inch conversion will be wrong here, so do a different trick:
        # We won't use this. We'll instead set card height by fixed inch blocks.

    plt.close(tmp_fig)

    # Compute per-card height in inches (compact & stable)
    card_h_in = (
        card_pad_top_in
        + 0.26  # title block
        + gap_title_sub_in
        + 0.20  # subtitle block
        + gap_sub_prompt_in
        + prompt_box_h_in
        + gap_prompt_answer_in
        + 0.18  # "Answer:" label block
        + gap_label_value_in
        + 0.22  # answer value block
        + card_pad_bot_in
    )

    fig_h = top_m_in + bot_m_in + n * card_h_in + (n - 1) * gap_in

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor=COLORS["background"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # inch<->axes conversion
    def x_in(x): return x / fig_w
    def y_in(y): return y / fig_h

    card_x0 = x_in(outer_lr_in)
    card_x1 = 1 - x_in(outer_lr_in)
    card_w = card_x1 - card_x0

    y_cursor_in = fig_h - top_m_in

    for i, s in enumerate(samples):
        case_id = s.get("case_id", "")
        prompt = str(s.get("prompt", ""))
        ans = str(s.get("answer", ""))

        y_top_in = y_cursor_in
        y_bot_in = y_top_in - card_h_in

        y_top = y_in(y_top_in)
        y_bot = y_in(y_bot_in)
        h = y_top - y_bot

        # Shadow
        shadow = FancyBboxPatch(
            (card_x0 + x_in(0.03), y_bot - y_in(0.03)),
            card_w, h,
            boxstyle="round,pad=0.010,rounding_size=0.018",
            linewidth=0, facecolor="#00000018", zorder=1
        )
        ax.add_patch(shadow)

        # Card
        card = FancyBboxPatch(
            (card_x0, y_bot),
            card_w, h,
            boxstyle="round,pad=0.010,rounding_size=0.018",
            linewidth=1.0, edgecolor=COLORS["border"], facecolor=COLORS["card"], zorder=2
        )
        ax.add_patch(card)

        # Accent bar
        accent = Rectangle(
            (card_x0 + x_in(accent_pad_in), y_bot + y_in(accent_pad_in)),
            x_in(accent_w_in), h - 2 * y_in(accent_pad_in),
            facecolor=COLORS["ioi"], zorder=3
        )
        ax.add_patch(accent)
        accent.set_clip_path(card)

        content_x = card_x0 + x_in(content_left_in)
        content_r = card_x1 - x_in(content_right_in)

        # Flow layout: y is in inches, convert at draw time
        y = y_top_in - card_pad_top_in

        # Title
        t = ax.text(content_x, y_in(y), f"Case {case_id}",
                    fontsize=fs_title, fontweight="bold",
                    va="top", ha="left", color=COLORS["text_primary"],
                    zorder=4, clip_on=True)
        t.set_clip_path(card)
        y -= (0.26 + gap_title_sub_in)

        # Subtitle
        t = ax.text(content_x, y_in(y), "Indirect Object Identification",
                    fontsize=fs_sub, style="italic",
                    va="top", ha="left", color=COLORS["text_secondary"],
                    zorder=4, clip_on=True)
        t.set_clip_path(card)
        y -= (0.20 + gap_sub_prompt_in)

        # Prompt box (fixed height)
        box_x0 = content_x - x_in(0.02)
        box_w  = (content_r - content_x) + x_in(0.04)
        box_h  = y_in(prompt_box_h_in)   # convert inch->axes via y_in
        box_y0 = y_in(y - prompt_box_h_in)

        # Fit text to the box (wrap + linespacing/fontsize adjustment)
        add_prompt_in_box(
            fig, ax,
            box_x0, box_y0, box_w, box_h,
            prompt,
            fontsize=fs_body,
            family=mono_family,
            color=COLORS["text_primary"],
            bg_color=COLORS["prompt_bg"],
            bg_alpha=0.65,
            pad_x=x_in(0.06),
            pad_y=y_in(0.08),
            linespacing_max=1.14,
            linespacing_min=0.98,
            clip_path=card
        )

        y -= (prompt_box_h_in + gap_prompt_answer_in)

        # Answer label
        t = ax.text(content_x, y_in(y), "Answer:",
                    fontsize=fs_sub, fontweight="bold",
                    va="top", ha="left", color=COLORS["text_secondary"],
                    zorder=4, clip_on=True)
        t.set_clip_path(card)
        y -= (0.18 + gap_label_value_in)

        # Answer value
        t = ax.text(content_x, y_in(y), ans,
                    fontsize=fs_ans, fontweight="bold",
                    va="top", ha="left", color=COLORS["answer_accent"],
                    zorder=4, clip_on=True)
        t.set_clip_path(card)

        # Advance cursor
        y_cursor_in = y_bot_in - gap_in

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight", pad_inches=0.02, dpi=dpi, facecolor=COLORS["background"])
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", type=Path, required=True, help="JSON/JSONL samples")
    ap.add_argument("--outfile", type=Path, required=True, help="e.g., fig/ioi-examples.pdf")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--wrap_width", type=int, default=40, help="hint only (kept for CLI compatibility)")
    ap.add_argument("--fig_w", type=float, default=3.35, help="inches; two-column ~3.3-3.5")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    samples = load_samples(args.infile)
    draw_cards(samples, args.outfile, n=args.n, wrap_width_hint=args.wrap_width, fig_w=args.fig_w, dpi=args.dpi)

if __name__ == "__main__":
    main()