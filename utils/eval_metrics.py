"""
Standardized evaluation metrics and population-based evaluation utilities.
All notebooks should use this module for evaluation consistency.
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_bis_metrics(bis_trajectory, target_bis=50.0):
    """Calculate performance metrics from a BIS trajectory.
    
    Args:
        bis_trajectory: Array of BIS values over time
        target_bis: Target BIS value (default 50.0)
    
    Returns:
        dict: Contains MDPE, MDAPE, Wobble, TimeInTarget
    """
    bis_trajectory = np.asarray(bis_trajectory, dtype=float)
    
    # Prediction Error (PE) as percentage
    pe = (bis_trajectory - target_bis) / target_bis * 100.0
    
    # Remove NaN values
    pe_clean = pe[np.isfinite(pe)]
    
    if len(pe_clean) == 0:
        return {
            'MDPE': np.nan,
            'MDAPE': np.nan,
            'Wobble': np.nan,
            'TimeInTarget': 0.0
        }
    
    # MDPE: Median Prediction Error
    mdpe = np.median(pe_clean)
    
    # MDAPE: Median Absolute Prediction Error
    mdape = np.median(np.abs(pe_clean))
    
    # Wobble: Median deviation from median PE
    wobble = np.median(np.abs(pe_clean - mdpe))
    
    # Time in Target: % of time within ±5 of target
    time_in_target = (np.abs(bis_trajectory - target_bis) <= 5).mean() * 100.0
    
    return {
        'MDPE': float(mdpe),
        'MDAPE': float(mdape),
        'Wobble': float(wobble),
        'TimeInTarget': float(time_in_target)
    }


def get_age_group(age, age_groups=None):
    """Map age to age group category.
    
    Args:
        age: Patient age in years
        age_groups: dict with format {'group_name': (min_age, max_age), ...}
                   Default uses standard groups: 25-29, 30-45, 46-60, 60-80, 80+
    
    Returns:
        str: Age group label
    """
    if age_groups is None:
        age_groups = {
            '25-29': (25, 29),
            '30-45': (30, 45),
            '46-60': (46, 60),
            '60-80': (60, 80),
            '80+': (80, 120)
        }
    
    age = int(age)
    for group_name, (min_age, max_age) in age_groups.items():
        if min_age <= age <= max_age:
            return group_name
    return 'Unknown'


def create_results_dataframe(patient_results, eval_lengths):
    """Convert simulation results to DataFrame format.
    
    Args:
        patient_results: list of dicts with keys:
            - patient_id
            - age
            - age_group
            - results: dict with key = episode_length (seconds), value = metrics dict
        eval_lengths: list of episode lengths (seconds)
    
    Returns:
        pd.DataFrame: One row per patient, columns for each metric/length combination
    """
    rows = []
    
    for result in patient_results:
        row = {
            'PatientID': result['patient_id'],
            'Age': result['age'],
            'AgeGroup': result['age_group']
        }
        
        # Add metrics for each episode length
        for ep_len in eval_lengths:
            if ep_len in result['results']:
                metrics = result['results'][ep_len]
                row[f'MDPE_{ep_len}s'] = metrics['MDPE']
                row[f'MDAPE_{ep_len}s'] = metrics['MDAPE']
                row[f'Wobble_{ep_len}s'] = metrics['Wobble']
                row[f'TimeInTarget_{ep_len}s'] = metrics['TimeInTarget']
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def create_summary_by_age_group(results_df, eval_lengths, age_groups=None):
    """Create summary statistics grouped by age group.
    
    Args:
        results_df: Output from create_results_dataframe()
        eval_lengths: list of episode lengths (seconds)
        age_groups: dict of age group definitions (optional)
    
    Returns:
        pd.DataFrame: One row per age group with mean/std of metrics
    """
    if age_groups is None:
        age_groups = {
            '25-29': (25, 29),
            '30-45': (30, 45),
            '46-60': (46, 60),
            '60-80': (60, 80),
            '80+': (80, 120)
        }
    
    summary_rows = []
    
    for group_name in age_groups.keys():
        group_df = results_df[results_df['AgeGroup'] == group_name]
        
        if len(group_df) == 0:
            continue
        
        row = {
            'AgeGroup': group_name,
            'N_Patients': len(group_df),
            'MeanAge': group_df['Age'].mean(),
            'StdAge': group_df['Age'].std()
        }
        
        # Calculate mean and std for each metric/length combo
        for ep_len in eval_lengths:
            for metric in ['MDPE', 'MDAPE', 'Wobble', 'TimeInTarget']:
                col_name = f'{metric}_{ep_len}s'
                if col_name in group_df.columns:
                    row[f'{metric}_{ep_len}s_mean'] = group_df[col_name].mean()
                    row[f'{metric}_{ep_len}s_std'] = group_df[col_name].std()
        
        summary_rows.append(row)
    
    return pd.DataFrame(summary_rows)


def save_evaluation_results(results_df, summary_df, algorithm_name, metrics_dir='metrics'):
    """Save evaluation results to JSON files.
    
    Args:
        results_df: Full results DataFrame (one row per patient)
        summary_df: Summary statistics DataFrame (one row per age group)
        algorithm_name: Name of algorithm (used for filename)
        metrics_dir: Directory to save JSON files to
    """
    metrics_path = Path(metrics_dir)
    metrics_path.mkdir(parents=True, exist_ok=True)
    
    # Save full results
    results_file = metrics_path / f'{algorithm_name}_results.json'
    results_df.to_json(results_file, orient='records', indent=2)
    print(f"Saved results to {results_file}")
    
    # Save summary
    summary_file = metrics_path / f'{algorithm_name}_summary.json'
    summary_df.to_json(summary_file, orient='records', indent=2)
    print(f"Saved summary to {summary_file}")


# ============================================================================
# EVALUATOR BASE CLASS
# ============================================================================

class EvaluatorBase:
    """Base class for algorithm-specific evaluators.
    
    Subclasses must implement:
    - _select_action(state) -> action_index
    - _step_environment(action) -> (bis, error, reward)
    """
    
    def __init__(self, target_bis=50.0, bis0=95.0, bis_max=75.0, ec50=3.5, hill=2.5):
        self.target_bis = target_bis
        self.bis0 = bis0
        self.bis_max = bis_max
        self.ec50 = ec50
        self.hill = hill
    
    def _select_action(self, state):
        """Select action from state. Must be overridden by subclass."""
        raise NotImplementedError("Subclass must implement _select_action()")
    
    def _step_environment(self, action):
        """Step environment and return (bis, error, reward). Must be overridden."""
        raise NotImplementedError("Subclass must implement _step_environment()")
    
    def simulate(self, duration_steps, initial_state=None):
        """Run simulation for fixed number of steps.
        
        Args:
            duration_steps: Number of timesteps to run
            initial_state: Initial state (algorithm-specific, optional)
        
        Returns:
            bis_log: Array of BIS values
            action_log: Array of actions taken
        """
        bis_log = []
        action_log = []
        
        # Reset environment/state (subclass-specific)
        if initial_state is not None:
            self._init_state(initial_state)
        else:
            self._init_state()
        
        for _ in range(duration_steps):
            state = self._get_current_state()
            action = self._select_action(state)
            bis, _, _ = self._step_environment(action)
            bis_log.append(bis)
            action_log.append(action)
        
        return np.array(bis_log), np.array(action_log)
    
    def _init_state(self, initial_state=None):
        """Initialize environment state. Subclass should override."""
        pass
    
    def _get_current_state(self):
        """Get current state from environment. Subclass should override."""
        pass
