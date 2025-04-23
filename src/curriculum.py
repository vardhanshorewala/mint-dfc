import math
import numpy as np
import torch
from typing import List, Dict, Optional


class Curriculum:
    def __init__(self, args):
        # args.dims and args.points each contain start, end, inc, interval attributes
        # inc denotes the change in n_dims,
        # this change is done every interval,
        # and start/end are the limits of the parameter
        self.n_dims_truncated = args.dims.start
        self.n_points = args.points.start
        self.n_dims_schedule = args.dims
        self.n_points_schedule = args.points
        self.step_count = 0

    def update(self):
        self.step_count += 1
        self.n_dims_truncated = self.update_var(
            self.n_dims_truncated, self.n_dims_schedule
        )
        self.n_points = self.update_var(self.n_points, self.n_points_schedule)

    def update_var(self, var, schedule):
        if self.step_count % schedule.interval == 0:
            var += schedule.inc

        return min(var, schedule.end)


# returns the final value of var after applying curriculum.
def get_final_var(init_var, total_steps, inc, n_steps, lim):
    final_var = init_var + math.floor((total_steps) / n_steps) * inc

    return min(final_var, lim)


class CurriculumManager:
    def __init__(self, config):
        """
        Initialize curriculum manager with configuration.

        Args:
            config: Configuration dictionary containing curriculum settings
        """
        self.config = config
        self.stages = config['training']['curriculum']['stages']
        self.current_stage = 0
        self.steps_in_stage = 0

    def get_current_distribution(self) -> Dict:
        """
        Get current function type distribution and weights.

        Returns:
            Dictionary containing function types and their weights
        """
        stage = self.stages[self.current_stage]
        return {
            'function_types': stage['functions'],
            'weights': stage.get('weights', None)  # None means uniform distribution
        }

    def should_advance_stage(self, metrics: Dict[str, float]) -> bool:
        """
        Check if we should advance to the next curriculum stage.

        Args:
            metrics: Dictionary of current performance metrics

        Returns:
            Boolean indicating whether to advance to next stage
        """
        if self.current_stage >= len(self.stages) - 1:
            return False

        stage = self.stages[self.current_stage]
        self.steps_in_stage += 1

        # Check if we've met both the performance threshold and minimum steps
        threshold_met = all(
            metrics.get(f"mse_{func}", float('inf')) < stage['threshold']
            for func in stage['functions']
        )
        steps_met = self.steps_in_stage >= stage['steps']

        return threshold_met and steps_met

    def advance_stage(self):
        """Advance to the next curriculum stage."""
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            self.steps_in_stage = 0

    def get_sampling_weights(self, batch_size: int) -> torch.Tensor:
        """
        Get sampling weights for the current batch.

        Args:
            batch_size: Size of the batch to generate

        Returns:
            Tensor of function type indices for each example in the batch
        """
        distribution = self.get_current_distribution()
        weights = distribution['weights']

        if weights is None:
            # Uniform distribution
            weights = np.ones(len(distribution['function_types'])) / len(distribution['function_types'])

        # Sample function types according to weights
        indices = np.random.choice(
            len(distribution['function_types']),
            size=batch_size,
            p=weights
        )
        return torch.from_numpy(indices)

    def get_active_functions(self) -> List[str]:
        """Get list of currently active function types."""
        return self.stages[self.current_stage]['functions']

    def get_progress(self) -> Dict:
        """Get current curriculum progress information."""
        return {
            'stage': self.current_stage,
            'total_stages': len(self.stages),
            'steps_in_stage': self.steps_in_stage,
            'active_functions': self.get_active_functions(),
            'distribution': self.get_current_distribution()
        }
