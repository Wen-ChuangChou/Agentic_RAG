from pathlib import Path
from utils.results_manager import load_evaluation_results
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_results(results_dir: str):
    """
    Read evaluation results from all JSON files in the specified directory.
    For each model & system type, count the proportion of scores 0, 1, 2.

    Args:
        results_dir: Directory to load the results from.

    Returns:
        List of dicts, each with:
          - model_name: str
          - system_type: str  (display name)
          - correct: float    (proportion of score==1)
          - partial: float    (proportion of score==0.5)
          - wrong: float      (proportion of score==0)
    """
    rows = []

    results_path = Path(results_dir)
    json_files = list(results_path.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {results_dir}")
        return rows

    # Mapping from internal keys to display labels
    system_display = {
        "agentic_rag": "Agentic RAG",
        "standard_rag": "Standard RAG",
        "standard": "Vanilla LLM",
    }

    for filepath in json_files:
        results = load_evaluation_results(results_dir, filepath.name)

        if "model_name" not in results:
            continue

        model_name = results["model_name"]

        for system_key, display_name in system_display.items():
            if system_key not in results:
                continue

            scores = results[system_key]['eval_score_LLM_judge_int']
            total = len(scores)
            if total == 0:
                continue

            # Count proportions (as percentages)
            correct_pct = (scores == 1).sum() / total * 100
            partial_pct = (scores == .5).sum() / total * 100
            wrong_pct = (scores == 0).sum() / total * 100

            rows.append({
                "model_name": model_name,
                "system_type": display_name,
                "Correct (2)": correct_pct,
                "Partially correct (1)": partial_pct,
                "Wrong (0)": wrong_pct,
            })

        print(f"Processed: {model_name}\n")

    return rows


def plot_stacked_bars(results_dir, rows):
    """
    Plot a stacked bar chart.
    Each bar = one (model, system_type) combination.
    Bottom segment = Correct, middle = Partially correct, top = Wrong.
    Bars are grouped by model, with one bar per system type.
    """
    if not rows:
        print("No scores to plot.")
        return

    df = pd.DataFrame(rows)

    # Determine unique models and system types (preserve insertion order)
    models = list(dict.fromkeys(df["model_name"]))
    system_types = list(dict.fromkeys(df["system_type"]))

    n_models = len(models)
    n_systems = len(system_types)

    # --- Style ---
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(max(8, n_models * 2.2), 6))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # Colors for the three score categories (bottom-to-top)
    # Correct = green-ish, Partial = gold, Wrong = red-ish
    color_correct = '#8faa82'
    color_partial = '#d9b66c'
    color_wrong = '#c56b46'

    bar_width = 0.22
    group_gap = 0.15  # extra space between model groups

    # Compute x positions for each bar
    x_positions = []
    x_ticks = []
    x_tick_labels = []
    bar_idx = 0

    for mi, model in enumerate(models):
        group_start = bar_idx
        for si, stype in enumerate(system_types):
            row = df[(df["model_name"] == model)
                     & (df["system_type"] == stype)]
            if row.empty:
                bar_idx += 1
                continue

            x = bar_idx * (bar_width + 0.05)
            correct_val = row["Correct (2)"].values[0]
            partial_val = row["Partially correct (1)"].values[0]
            wrong_val = row["Wrong (0)"].values[0]

            # Stack: bottom = correct, middle = partial, top = wrong
            ax.bar(x,
                   correct_val,
                   width=bar_width,
                   color=color_correct,
                   edgecolor='white',
                   linewidth=0.8,
                   label='Correct (2)' if mi == 0 and si == 0 else '')
            ax.bar(
                x,
                partial_val,
                width=bar_width,
                bottom=correct_val,
                color=color_partial,
                edgecolor='white',
                linewidth=0.8,
                label='Partially correct (1)' if mi == 0 and si == 0 else '')
            ax.bar(x,
                   wrong_val,
                   width=bar_width,
                   bottom=correct_val + partial_val,
                   color=color_wrong,
                   edgecolor='white',
                   linewidth=0.8,
                   label='Wrong (0)' if mi == 0 and si == 0 else '')

            # Put system-type label below the bar
            x_ticks.append(x)
            x_tick_labels.append(stype)

            bar_idx += 1

        # Add a model-name annotation centred under its group
        group_end = bar_idx - 1
        group_center = ((group_start) * (bar_width + 0.05) + (group_end) *
                        (bar_width + 0.05)) / 2
        ax.text(group_center,
                -18,
                model,
                ha='center',
                va='top',
                fontsize=11,
                color='white',
                fontweight='bold')

        bar_idx += 1  # extra gap between model groups

    # X-axis: system-type labels (rotated)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels,
                       rotation=45,
                       ha='right',
                       fontsize=10,
                       color='white')

    # Y-axis
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'],
                       color='white',
                       fontsize=11)
    ax.tick_params(axis='y', left=False)

    # Add percentage labels inside each segment (if large enough)
    for container in ax.containers:
        for bar in container:
            height = bar.get_height()
            if height > 4:  # only label segments tall enough to fit text
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + height / 2,
                        f'{height:.0f}%',
                        ha='center',
                        va='center',
                        fontsize=8,
                        color='white',
                        fontweight='bold')

    # Title
    ax.set_title("Score distribution for technical Q&A",
                 fontsize=14,
                 color='white',
                 pad=25)

    # Hide spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    # Reverse so that "Wrong (0)" appears first (top of legend matches top of bar)
    legend = ax.legend(handles[::-1],
                       labels[::-1],
                       loc='upper right',
                       frameon=False,
                       fontsize=11)
    plt.setp(legend.get_texts(), color='white')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)  # room for model names

    output_path = Path(results_dir) / "score_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to {output_path}")
    plt.show()


def main():
    """
    Main function to load and show evaluation results.
    """
    parser = argparse.ArgumentParser(
        description="Visualize score distribution as stacked bar chart.")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    rows = read_results(args.results_dir)
    plot_stacked_bars(args.results_dir, rows)


if __name__ == "__main__":
    main()
