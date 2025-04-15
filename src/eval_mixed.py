import os
import yaml
import torch
import wandb
from argparse import ArgumentParser

from eval import get_run_metrics
from plot_utils import plot_mixed_function_metrics, collect_mixed_function_results
from models import build_model
from curriculum import CurriculumManager

def main():
    parser = ArgumentParser()
    parser.add_argument('--run_dir', required=True, help='Directory containing the trained model')
    parser.add_argument('--config', required=True, help='Path to config file used for training')
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Initialize curriculum manager
    curriculum = CurriculumManager(config)

    # Load model
    model = build_model(config.model)
    checkpoint = torch.load(os.path.join(args.run_dir, 'model.pt'))
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # Get metrics
    metrics = get_run_metrics(args.run_dir)

    # Plot results
    function_types = config['training']['task_kwargs']['function_types']
    fig, axes = plot_mixed_function_metrics(metrics, function_types)

    # Save plots
    os.makedirs(os.path.join(args.run_dir, 'plots'), exist_ok=True)
    fig.savefig(os.path.join(args.run_dir, 'plots', 'mixed_function_metrics.png'))

    # Log to wandb if enabled
    if config.get('wandb', {}).get('enabled', False):
        wandb.init(
            project=config['wandb'].get('project', 'mixed-function-learning'),
            name=config['wandb'].get('name', 'evaluation'),
            config=config
        )
        wandb.log({
            'metrics': metrics,
            'plots': wandb.Image(fig)
        })

if __name__ == '__main__':
    main()