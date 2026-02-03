#!/usr/bin/env python3
"""
Generate visualizations for multiple representative samples.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
from pathlib import Path

# Select representative samples to visualize
SAMPLES_TO_VISUALIZE = [
    2,   # High exact matches (2/6), short (18 tokens)
    5,   # Short sequence (15 tokens), 1 exact
    10,  # Medium (21 tokens), 0 exact, 3 top1
    13,  # Very short (17 tokens)
    14,  # Long (24 tokens), 1 exact
    17,  # No exact matches, very long (26 tokens)
    25,  # High exact matches (2/6), short (16 tokens)
    30,  # Medium (20 tokens), 0 exact
    57,  # High exact matches (2/6), medium (22 tokens)
    60,  # Short (18 tokens), 1 exact
    75,  # Medium (21 tokens), 1 exact
    85,  # Medium (22 tokens), 0 exact
    99,  # Original sample (22 tokens)
]

# Load data
project_root = Path(__file__).parent.parent.parent
with open(project_root / 'results/1-order/topk_comparison_Llama-3.2-3B-Instruct.json', 'r') as f:
    data = json.load(f)

# Clean token display
def clean_token(t):
    t = t.replace('Ġ', ' ').replace('▁', ' ')
    if t == '<|begin_of_text|>':
        return '<BOS>'
    return t

# Method display config
METHOD_CONFIG = {
    'input_causal': {'name': 'Input Causal', 'color': '#FF6B9D', 'dark': '#C2185B'},
    'depass': {'name': 'DePass', 'color': '#FF5252', 'dark': '#D32F2F'},
    'shapley': {'name': 'Shapley', 'color': '#00BFA5', 'dark': '#00796B'},
    'grad_input': {'name': 'Grad×Input', 'color': '#7C4DFF', 'dark': '#5E35B1'},
    'grad_sam': {'name': 'GradSAM', 'color': '#FFB300', 'dark': '#F57C00'},
    'attention_rollout': {'name': 'Attn Rollout', 'color': '#2979FF', 'dark': '#1565C0'},
}

def draw_token_row(ax, tokens_list, start_idx, y_center, greedy_top2, box_height=0.16):
    """Draw a row of tokens with enhanced visual effects and more spacing."""
    n = len(tokens_list)

    # Calculate widths - more generous spacing
    widths = []
    for t in tokens_list:
        clean_t = clean_token(t)
        w = max(len(clean_t) * 0.020, 0.042)
        widths.append(w)

    total_width = sum(widths) + 0.014 * (n - 1)
    start_x = (1 - total_width) / 2

    x = start_x
    for i, (token, width) in enumerate(zip(tokens_list, widths)):
        global_idx = start_idx + i
        clean_t = clean_token(token)

        # Style based on importance - darker colors
        if global_idx == greedy_top2[0]:
            facecolor = '#FFCDD2'
            edgecolor = '#E53935'
            linewidth = 5
            fontweight = 'bold'
            fontcolor = '#8B0000'  # Much darker red
            shadow = True
            glow_color = '#FFEBEE'
        elif global_idx == greedy_top2[1]:
            facecolor = '#FFE0B2'
            edgecolor = '#FB8C00'
            linewidth = 5
            fontweight = 'bold'
            fontcolor = '#BF360C'  # Much darker orange
            shadow = True
            glow_color = '#FFF3E0'
        else:
            facecolor = '#FAFAFA'
            edgecolor = '#CFD8DC'
            linewidth = 2
            fontweight = 'bold'  # Make all tokens bold
            fontcolor = '#212121'  # Much darker gray
            shadow = False
            glow_color = None

        # Draw shadow/glow for important tokens - larger glow
        if shadow:
            for offset in [0.008, 0.006, 0.004]:
                shadow_rect = FancyBboxPatch(
                    (x + offset, y_center - box_height/2 - offset), width, box_height,
                    boxstyle="round,pad=0.014,rounding_size=0.03",
                    facecolor=glow_color, edgecolor='none',
                    alpha=0.35,
                    transform=ax.transAxes, zorder=1)
                ax.add_patch(shadow_rect)

        # Draw main box
        rect = FancyBboxPatch(
            (x, y_center - box_height/2), width, box_height,
            boxstyle="round,pad=0.014,rounding_size=0.03",
            facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth,
            transform=ax.transAxes, zorder=2)
        ax.add_patch(rect)

        # Token text - larger and bolder
        ax.text(x + width/2, y_center + 0.015, clean_t,
               ha='center', va='center', fontsize=19, fontweight=fontweight,
               color=fontcolor, transform=ax.transAxes, fontfamily='monospace',
               zorder=3)

        # Index below with badge for important tokens - larger
        if global_idx in greedy_top2:
            # Highlight index
            ax.text(x + width/2, y_center - box_height/2 - 0.05, f'{global_idx}',
                   ha='center', va='top', fontsize=15, color=fontcolor,
                   fontweight='bold', transform=ax.transAxes)
        else:
            ax.text(x + width/2, y_center - box_height/2 - 0.05, f'{global_idx}',
                   ha='center', va='top', fontsize=13, color='#424242',
                   fontweight='bold', transform=ax.transAxes)

        x += width + 0.014


def visualize_sample(sample_data, output_path):
    """Generate visualization for a single sample."""

    tokens = sample_data['tokens']
    greedy_top2 = sample_data['greedy_top2']
    method_results = sample_data['method_results']

    # Create figure
    fig = plt.figure(figsize=(20, 32))
    fig.patch.set_facecolor('#F8F9FA')

    # Main title
    fig.text(0.5, 0.985, f'Token Attribution Analysis: Sample {sample_data["sample_idx"]}',
             ha='center', va='top', fontsize=32, fontweight='bold', color='#000000')
    fig.text(0.5, 0.970, 'Comparing Attribution Methods Against Greedy Optimal Baseline',
             ha='center', va='top', fontsize=24, color='#1A1A1A', style='italic', fontweight='600')

    # ============================================================================
    # Part 1: Token sequence display
    # ============================================================================

    ax_tokens = fig.add_axes([0.03, 0.60, 0.94, 0.34])
    ax_tokens.set_xlim(0, 1)
    ax_tokens.set_ylim(0, 1)
    ax_tokens.axis('off')

    # Add subtle background panel
    ax_tokens.add_patch(Rectangle(
        (0.01, 0.01), 0.98, 0.98,
        facecolor='white', edgecolor='#BDBDBD', linewidth=2.5,
        transform=ax_tokens.transAxes, zorder=0))

    # Section title
    title_bg = FancyBboxPatch(
        (0.20, 0.90), 0.60, 0.08,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor='#1A1A2E', edgecolor='none',
        transform=ax_tokens.transAxes, zorder=1)
    ax_tokens.add_patch(title_bg)

    ax_tokens.text(0.5, 0.940, 'Ground Truth (Greedy Optimal Selection)',
                  ha='center', va='center', fontsize=22, fontweight='bold',
                  color='white', transform=ax_tokens.transAxes, zorder=2)

    # Enhanced legend
    legend_y = 0.80
    # Top-1 badge
    ax_tokens.add_patch(FancyBboxPatch(
        (0.28, legend_y - 0.030), 0.065, 0.060,
        boxstyle="round,pad=0.008,rounding_size=0.02",
        facecolor='#FFCDD2', edgecolor='#E53935', linewidth=3.5,
        transform=ax_tokens.transAxes, zorder=1))
    ax_tokens.text(0.3125, legend_y, '1st', ha='center', va='center', fontsize=16,
                  fontweight='bold', color='#B71C1C', transform=ax_tokens.transAxes, zorder=2)
    ax_tokens.text(0.355, legend_y, 'Most Important', ha='left', va='center', fontsize=17,
                  fontweight='bold', color='#B71C1C', transform=ax_tokens.transAxes)

    # Top-2 badge
    ax_tokens.add_patch(FancyBboxPatch(
        (0.52, legend_y - 0.030), 0.065, 0.060,
        boxstyle="round,pad=0.008,rounding_size=0.02",
        facecolor='#FFE0B2', edgecolor='#FB8C00', linewidth=3.5,
        transform=ax_tokens.transAxes, zorder=1))
    ax_tokens.text(0.5525, legend_y, '2nd', ha='center', va='center', fontsize=16,
                  fontweight='bold', color='#E65100', transform=ax_tokens.transAxes, zorder=2)
    ax_tokens.text(0.595, legend_y, 'Second Important', ha='left', va='center', fontsize=17,
                  fontweight='bold', color='#E65100', transform=ax_tokens.transAxes)

    # Split tokens into rows (11 tokens per row)
    num_tokens = len(tokens)
    tokens_per_row = 11
    num_rows = (num_tokens + tokens_per_row - 1) // tokens_per_row

    # Calculate y positions for rows
    start_y = 0.62
    row_spacing = 0.36 / (num_rows + 1)

    for row_idx in range(num_rows):
        start_idx = row_idx * tokens_per_row
        end_idx = min(start_idx + tokens_per_row, num_tokens)
        row_tokens = tokens[start_idx:end_idx]
        y_pos = start_y - row_idx * row_spacing
        draw_token_row(ax_tokens, row_tokens, start_idx, y_pos, greedy_top2)

    # Prompt text at bottom
    prompt_bg = FancyBboxPatch(
        (0.06, 0.04), 0.88, 0.10,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor='#ECEFF1', edgecolor='#B0BEC5', linewidth=2,
        transform=ax_tokens.transAxes, zorder=0)
    ax_tokens.add_patch(prompt_bg)

    prompt_text = f'Prompt: "{sample_data["prompt"]}"  →  Answer: {sample_data["answer"]}'
    ax_tokens.text(0.5, 0.09, prompt_text, ha='center', va='center', fontsize=16,
                  style='italic', color='#1A1A1A', transform=ax_tokens.transAxes,
                  fontweight='bold', zorder=1)

    # ============================================================================
    # Part 2: Method comparison
    # ============================================================================

    ax_table = fig.add_axes([0.03, 0.04, 0.94, 0.52])
    ax_table.set_xlim(0, 1)
    ax_table.set_ylim(0, 1)
    ax_table.axis('off')

    # Add background panel
    ax_table.add_patch(Rectangle(
        (0.01, 0.01), 0.98, 0.98,
        facecolor='white', edgecolor='#BDBDBD', linewidth=2.5,
        transform=ax_table.transAxes, zorder=0))

    # Section title
    title_bg = FancyBboxPatch(
        (0.28, 0.925), 0.44, 0.065,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor='#1A1A2E', edgecolor='none',
        transform=ax_table.transAxes, zorder=1)
    ax_table.add_patch(title_bg)

    ax_table.text(0.5, 0.9575, 'Attribution Methods Comparison',
                 ha='center', va='center', fontsize=22, fontweight='bold',
                 color='white', transform=ax_table.transAxes, zorder=2)

    # Table header
    header_y = 0.855
    header_rect = Rectangle(
        (0.06, header_y - 0.040), 0.88, 0.065,
        facecolor='#ECEFF1', edgecolor='#90A4AE', linewidth=2.5,
        transform=ax_table.transAxes, zorder=0)
    ax_table.add_patch(header_rect)

    headers = [('Method', 0.18), ('Top-1 Token', 0.42), ('Top-2 Token', 0.64), ('Match Status', 0.85)]
    for text, x in headers:
        ax_table.text(x, header_y, text, ha='center', va='center', fontsize=18,
                     fontweight='bold', color='#000000', transform=ax_table.transAxes)

    # Methods
    methods = ['input_causal', 'depass', 'shapley', 'grad_input', 'grad_sam', 'attention_rollout']
    row_height = 0.112
    start_y = 0.765

    for i, method in enumerate(methods):
        cfg = METHOD_CONFIG[method]
        result = method_results[method]
        top2 = result['top2']
        y = start_y - i * row_height

        # Alternating background
        if i % 2 == 0:
            ax_table.add_patch(Rectangle(
                (0.06, y - row_height/2 + 0.012), 0.88, row_height - 0.018,
                facecolor='#F5F5F5', edgecolor='#E0E0E0', linewidth=0.5,
                transform=ax_table.transAxes, zorder=0))

        # Method name with badge
        badge_size = 0.016
        ax_table.add_patch(mpatches.Circle(
            (0.085, y), badge_size * 2, facecolor=cfg['color'], alpha=0.2,
            edgecolor='none', transform=ax_table.transAxes, zorder=1))
        ax_table.add_patch(mpatches.Circle(
            (0.085, y), badge_size, facecolor=cfg['color'],
            edgecolor=cfg['dark'], linewidth=2.5,
            transform=ax_table.transAxes, zorder=2))

        ax_table.text(0.11, y, cfg['name'], ha='left', va='center', fontsize=18,
                     fontweight='bold', color='#000000', transform=ax_table.transAxes)

        # Top-1 token
        t1_idx = top2[0]
        t1_match = (t1_idx == greedy_top2[0])

        if t1_match:
            t1_bg_color = '#E8F5E9'
            t1_color = '#1B5E20'
            t1_symbol = '✓'
            t1_edge = '#4CAF50'
        else:
            t1_bg_color = '#FFEBEE'
            t1_color = '#B71C1C'
            t1_symbol = '✗'
            t1_edge = '#EF5350'

        t1_width = 0.17
        ax_table.add_patch(FancyBboxPatch(
            (0.42 - t1_width/2, y - 0.038), t1_width, 0.076,
            boxstyle="round,pad=0.006,rounding_size=0.018",
            facecolor=t1_bg_color, edgecolor=t1_edge, linewidth=2,
            transform=ax_table.transAxes, zorder=1))

        ax_table.text(0.345, y, f'[{t1_idx}]', ha='right', va='center', fontsize=15,
                     color=t1_color, fontfamily='monospace', fontweight='bold',
                     transform=ax_table.transAxes, zorder=2)
        ax_table.text(0.425, y, clean_token(tokens[t1_idx]), ha='center', va='center',
                     fontsize=16, color=t1_color, fontfamily='monospace', fontweight='bold',
                     transform=ax_table.transAxes, zorder=2)
        ax_table.text(0.495, y, t1_symbol, ha='left', va='center', fontsize=19,
                     color=t1_color, fontweight='bold', transform=ax_table.transAxes, zorder=2)

        # Top-2 token
        t2_idx = top2[1]
        t2_match = (t2_idx == greedy_top2[1])

        if t2_match:
            t2_bg_color = '#E8F5E9'
            t2_color = '#1B5E20'
            t2_symbol = '✓'
            t2_edge = '#4CAF50'
        else:
            t2_bg_color = '#FFEBEE'
            t2_color = '#B71C1C'
            t2_symbol = '✗'
            t2_edge = '#EF5350'

        t2_width = 0.17
        ax_table.add_patch(FancyBboxPatch(
            (0.64 - t2_width/2, y - 0.038), t2_width, 0.076,
            boxstyle="round,pad=0.006,rounding_size=0.018",
            facecolor=t2_bg_color, edgecolor=t2_edge, linewidth=2,
            transform=ax_table.transAxes, zorder=1))

        ax_table.text(0.565, y, f'[{t2_idx}]', ha='right', va='center', fontsize=15,
                     color=t2_color, fontfamily='monospace', fontweight='bold',
                     transform=ax_table.transAxes, zorder=2)
        ax_table.text(0.645, y, clean_token(tokens[t2_idx]), ha='center', va='center',
                     fontsize=16, color=t2_color, fontfamily='monospace', fontweight='bold',
                     transform=ax_table.transAxes, zorder=2)
        ax_table.text(0.715, y, t2_symbol, ha='left', va='center', fontsize=19,
                     color=t2_color, fontweight='bold', transform=ax_table.transAxes, zorder=2)

        # Match status badge
        exact = result['exact_match']
        set_m = result['set_match']
        top1_m = result['top1_match']

        if exact:
            status, bg, fg, icon = "EXACT", '#C8E6C9', '#1B5E20', '✓✓'
        elif set_m:
            status, bg, fg, icon = "SET", '#BBDEFB', '#0D47A1', '✓'
        elif top1_m:
            status, bg, fg, icon = "TOP-1", '#FFF9C4', '#F57F17', '½'
        else:
            status, bg, fg, icon = "NONE", '#FFCDD2', '#B71C1C', '✗'

        badge_w = 0.11
        ax_table.add_patch(FancyBboxPatch(
            (0.85 - badge_w/2 + 0.004, y - 0.042), badge_w, 0.084,
            boxstyle="round,pad=0.006,rounding_size=0.022",
            facecolor='#00000020', edgecolor='none',
            transform=ax_table.transAxes, zorder=1))

        ax_table.add_patch(FancyBboxPatch(
            (0.85 - badge_w/2, y - 0.044), badge_w, 0.088,
            boxstyle="round,pad=0.006,rounding_size=0.022",
            facecolor=bg, edgecolor=fg, linewidth=3,
            transform=ax_table.transAxes, zorder=2))

        ax_table.text(0.810, y, icon, ha='center', va='center', fontsize=16,
                     fontweight='bold', color=fg, transform=ax_table.transAxes, zorder=3)
        ax_table.text(0.870, y, status, ha='center', va='center', fontsize=15,
                     fontweight='bold', color=fg, transform=ax_table.transAxes, zorder=3)

    # Bottom border
    bottom_y = start_y - len(methods) * row_height + 0.040
    ax_table.plot([0.06, 0.94], [bottom_y, bottom_y],
                 color='#90A4AE', linewidth=2.5, transform=ax_table.transAxes)

    # Ground truth reference
    ref_y = bottom_y - 0.075
    ref_bg = FancyBboxPatch(
        (0.13, ref_y - 0.035), 0.74, 0.075,
        boxstyle="round,pad=0.012,rounding_size=0.022",
        facecolor='#ECEFF1', edgecolor='#78909C', linewidth=2.5,
        transform=ax_table.transAxes, zorder=0)
    ax_table.add_patch(ref_bg)

    ax_table.text(0.165, ref_y + 0.003, 'Ground Truth:', ha='left', va='center',
                 fontsize=17, color='#000000', fontweight='bold',
                 transform=ax_table.transAxes, zorder=1)

    # Top-1 token reference
    ax_table.add_patch(FancyBboxPatch(
        (0.31, ref_y - 0.024), 0.15, 0.052,
        boxstyle="round,pad=0.006,rounding_size=0.018",
        facecolor='#FFCDD2', edgecolor='#E53935', linewidth=2.5,
        transform=ax_table.transAxes, zorder=1))
    ax_table.text(0.315, ref_y + 0.003, f'[{greedy_top2[0]}]', ha='left', va='center',
                 fontsize=15, color='#8B0000', fontfamily='monospace', fontweight='bold',
                 transform=ax_table.transAxes, zorder=2)
    ax_table.text(0.385, ref_y + 0.003, clean_token(tokens[greedy_top2[0]]),
                 ha='center', va='center', fontsize=16, color='#8B0000',
                 fontfamily='monospace', fontweight='bold',
                 transform=ax_table.transAxes, zorder=2)

    ax_table.text(0.49, ref_y + 0.003, '•', ha='center', va='center',
                 fontsize=20, color='#424242', transform=ax_table.transAxes, zorder=1)

    # Top-2 token reference
    ax_table.add_patch(FancyBboxPatch(
        (0.53, ref_y - 0.024), 0.15, 0.052,
        boxstyle="round,pad=0.006,rounding_size=0.018",
        facecolor='#FFE0B2', edgecolor='#FB8C00', linewidth=2.5,
        transform=ax_table.transAxes, zorder=1))
    ax_table.text(0.535, ref_y + 0.003, f'[{greedy_top2[1]}]', ha='left', va='center',
                 fontsize=15, color='#BF360C', fontfamily='monospace', fontweight='bold',
                 transform=ax_table.transAxes, zorder=2)
    ax_table.text(0.605, ref_y + 0.003, clean_token(tokens[greedy_top2[1]]),
                 ha='center', va='center', fontsize=16, color='#BF360C',
                 fontfamily='monospace', fontweight='bold',
                 transform=ax_table.transAxes, zorder=2)

    # Legend for match status
    legend_y = bottom_y - 0.165
    ax_table.text(0.23, legend_y, 'Match Legend:', ha='right', va='center',
                 fontsize=15, color='#000000', fontweight='bold',
                 transform=ax_table.transAxes)

    match_legends = [
        ('EXACT', '#C8E6C9', '#1B5E20', 0.28),
        ('SET', '#BBDEFB', '#0D47A1', 0.42),
        ('TOP-1', '#FFF9C4', '#F57F17', 0.56),
        ('NONE', '#FFCDD2', '#B71C1C', 0.70),
    ]

    for status, bg, fg, x_pos in match_legends:
        ax_table.add_patch(FancyBboxPatch(
            (x_pos - 0.040, legend_y - 0.018), 0.080, 0.036,
            boxstyle="round,pad=0.004,rounding_size=0.018",
            facecolor=bg, edgecolor=fg, linewidth=2,
            transform=ax_table.transAxes, zorder=1))
        ax_table.text(x_pos, legend_y, status, ha='center', va='center',
                     fontsize=13, fontweight='bold', color=fg,
                     transform=ax_table.transAxes, zorder=2)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#F8F9FA',
               edgecolor='none', pad_inches=0.2)
    plt.close(fig)

    print(f"✓ Saved: {output_path.name}")


# Generate visualizations for selected samples
output_dir = project_root / 'results/1-order/visualizations'

for sample_idx in SAMPLES_TO_VISUALIZE:
    sample = next(s for s in data['samples'] if s['sample_idx'] == sample_idx)
    output_path = output_dir / f'sample_{sample_idx:03d}_enhanced.png'
    visualize_sample(sample, output_path)

print(f"\n✓ Generated {len(SAMPLES_TO_VISUALIZE)} visualizations in {output_dir}")
