import os

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from eval import get_run_metrics, baseline_names, get_model_from_run
from models import build_model

sns.set_theme("notebook", "darkgrid")
palette = sns.color_palette("colorblind")


relevant_model_names = {
    "linear_regression": [
        "Transformer",
        "Least Squares",
        "3-Nearest Neighbors",
        "Averaging",
    ],
    "sparse_linear_regression": [
        "Transformer",
        "Least Squares",
        "3-Nearest Neighbors",
        "Averaging",
        "Lasso (alpha=0.01)",
    ],
    "decision_tree": [
        "Transformer",
        "3-Nearest Neighbors",
        "2-layer NN, GD",
        "Greedy Tree Learning",
        "XGBoost",
    ],
    "relu_2nn_regression": [
        "Transformer",
        "Least Squares",
        "3-Nearest Neighbors",
        "2-layer NN, GD",
    ],
    "mixed_function": [
        "Transformer",
        "Least Squares",
        "3-Nearest Neighbors",
    ],
}


def basic_plot(metrics, models=None, trivial=1.0):
    fig, ax = plt.subplots(1, 1)

    if models is not None:
        metrics = {k: metrics[k] for k in models}

    color = 0
    ax.axhline(trivial, ls="--", color="gray")
    for name, vs in metrics.items():
        ax.plot(vs["mean"], "-", label=name, color=palette[color % 10], lw=2)
        low = vs["bootstrap_low"]
        high = vs["bootstrap_high"]
        ax.fill_between(range(len(low)), low, high, alpha=0.3)
        color += 1
    ax.set_xlabel("in-context examples")
    ax.set_ylabel("squared error")
    ax.set_xlim(-1, len(low) + 0.1)
    ax.set_ylim(-0.1, 1.25)

    legend = ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    fig.set_size_inches(4, 3)
    for line in legend.get_lines():
        line.set_linewidth(3)

    return fig, ax


def collect_results(run_dir, df, valid_row=None, rename_eval=None, rename_model=None):
    all_metrics = {}
    for _, r in df.iterrows():
        if valid_row is not None and not valid_row(r):
            continue

        run_path = os.path.join(run_dir, r.task, r.run_id)
        _, conf = get_model_from_run(run_path, only_conf=True)

        print(r.run_name, r.run_id)
        metrics = get_run_metrics(run_path, skip_model_load=True)

        for eval_name, results in sorted(metrics.items()):
            processed_results = {}
            for model_name, m in results.items():
                if "gpt2" in model_name in model_name:
                    model_name = r.model
                    if rename_model is not None:
                        model_name = rename_model(model_name, r)
                else:
                    model_name = baseline_names(model_name)
                m_processed = {}
                n_dims = conf.model.n_dims

                xlim = 2 * n_dims + 1
                if r.task in ["relu_2nn_regression", "decision_tree"]:
                    xlim = 200

                normalization = n_dims
                if r.task == "sparse_linear_regression":
                    normalization = int(r.kwargs.split("=")[-1])
                if r.task == "decision_tree":
                    normalization = 1

                for k, v in m.items():
                    v = v[:xlim]
                    v = [vv / normalization for vv in v]
                    m_processed[k] = v
                processed_results[model_name] = m_processed
            if rename_eval is not None:
                eval_name = rename_eval(eval_name, r)
            if eval_name not in all_metrics:
                all_metrics[eval_name] = {}
            all_metrics[eval_name].update(processed_results)
    return all_metrics


def plot_mixed_function_metrics(metrics, function_types):
    """Plot metrics for mixed function experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Plot per-class MSE
    for i, func_type in enumerate(function_types):
        mse = metrics[f'mse_{func_type}']
        axes[0].plot(mse['mean'], label=func_type, color=palette[i])
        axes[0].fill_between(
            range(len(mse['mean'])),
            mse['bootstrap_low'],
            mse['bootstrap_high'],
            alpha=0.3
        )
    axes[0].set_title('Per-Class MSE')
    axes[0].set_xlabel('In-context examples')
    axes[0].set_ylabel('MSE')
    axes[0].legend()

    # Plot adaptation speed
    adaptation = metrics['adaptation_speed']
    axes[1].plot(adaptation['steps'], adaptation['error'], label='Average Error')
    axes[1].set_title('Adaptation Speed')
    axes[1].set_xlabel('Training Steps')
    axes[1].set_ylabel('Error')

    # Plot class distribution
    if 'class_distribution' in metrics:
        dist = metrics['class_distribution']
        axes[2].bar(range(len(function_types)), dist, tick_label=function_types)
        axes[2].set_title('Class Distribution')
        axes[2].set_ylabel('Frequency')

    # Plot confusion matrix if available
    if 'class_confusion' in metrics:
        conf_mat = metrics['class_confusion']
        sns.heatmap(conf_mat, ax=axes[3], xticklabels=function_types,
                   yticklabels=function_types, annot=True, fmt='.2f')
        axes[3].set_title('Class Confusion Matrix')

    plt.tight_layout()
    return fig, axes


def collect_mixed_function_results(run_dir, df, curriculum_manager=None):
    """Collect results specifically for mixed function experiments."""
    metrics = collect_results(run_dir, df)

    # Add mixed function specific metrics
    for eval_name, results in metrics.items():
        if curriculum_manager is not None:
            active_funcs = curriculum_manager.get_active_functions()
            results['active_functions'] = active_funcs
            results['curriculum_stage'] = curriculum_manager.get_progress()

    return metrics
