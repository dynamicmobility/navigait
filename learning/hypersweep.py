"""
Hyperparameter Sweep for RL Training

This module provides functionality for running hyperparameter sweeps
across PPO training configurations for bipedal robot control.

Usage:
    python -m learning.hypersweep config/bruce-canonical.yaml --method grid
    python -m learning.hypersweep config/bruce-canonical.yaml --method random --n-trials 50
    python -m learning.hypersweep config/bruce-canonical.yaml --method optuna --n-trials 100
"""

import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=true"

import argparse
import copy
import datetime
import itertools
import json
import random
import yaml
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

# Internal imports
from utils.setupGPU import run_setup
from learning.startup import read_config, create_environment, get_commit_hash
from learning.training import setup_training, train


# ============================================================================
# Hyperparameter Search Space Definition
# ============================================================================

# Define the search space for hyperparameters
# Each entry is: (type, values/range)
# type can be: 'categorical', 'uniform', 'loguniform', 'int_uniform'

SEARCH_SPACE = {
    # PPO Parameters
    'ppo_params.learning_rate': ('loguniform', 1e-5, 1e-2),
    'ppo_params.batch_size': ('categorical', [128, 256, 512, 1024]),
    'ppo_params.clipping_epsilon': ('uniform', 0.1, 0.3),
    'ppo_params.entropy_cost': ('loguniform', 1e-4, 1e-1),
    'ppo_params.discounting': ('uniform', 0.95, 0.995),
    'ppo_params.num_updates_per_batch': ('categorical', [2, 4, 8]),
    'ppo_params.unroll_length': ('categorical', [16, 32, 64]),
    'ppo_params.num_minibatches': ('categorical', [16, 32, 64]),
    'ppo_params.max_grad_norm': ('uniform', 0.5, 2.0),
    
    # Network Architecture
    'network_params.policy_hidden_layer_sizes': ('categorical', [
        [128, 64],
        [256, 128],
        [256, 128, 64],
        [512, 256, 128],
        [256, 256, 256],
    ]),
    'network_params.value_hidden_layer_sizes': ('categorical', [
        [128, 64],
        [256, 128],
        [256, 128, 64],
        [512, 256, 128],
        [256, 256, 256],
    ]),
    
    # Environment Parameters
    'env_config.action_scale': ('uniform', 0.2, 0.5),
    'env_config.noise_scale': ('uniform', 0.5, 2.0),
    'env_config.gait_freq': ('uniform', 1.0, 2.0),
    
    # Reward Weights (key ones to tune)
    'env_config.reward.weights.linvel_tracking': ('uniform', 1.0, 5.0),
    'env_config.reward.weights.angvel_tracking': ('uniform', 1.0, 5.0),
    'env_config.reward.weights.action_rate': ('uniform', -2.0, -0.5),
    'env_config.reward.weights.jt_imitation': ('uniform', 0.5, 2.0),
}

# Reduced search space for quick experiments
SEARCH_SPACE_MINIMAL = {
    'ppo_params.learning_rate': ('loguniform', 1e-4, 1e-2),
    'ppo_params.batch_size': ('categorical', [256, 512]),
    'ppo_params.entropy_cost': ('loguniform', 1e-3, 1e-1),
    'ppo_params.discounting': ('uniform', 0.95, 0.99),
    'network_params.policy_hidden_layer_sizes': ('categorical', [
        [256, 128, 64],
        [512, 256, 128],
    ]),
}


# ============================================================================
# Utility Functions
# ============================================================================

def set_nested_value(d: dict, key_path: str, value: Any) -> None:
    """Set a value in a nested dictionary using dot notation.
    
    Args:
        d: Dictionary to modify
        key_path: Dot-separated path (e.g., 'ppo_params.learning_rate')
        value: Value to set
    """
    keys = key_path.split('.')
    current = d
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def get_nested_value(d: dict, key_path: str) -> Any:
    """Get a value from a nested dictionary using dot notation."""
    keys = key_path.split('.')
    current = d
    for key in keys:
        current = current[key]
    return current


def sample_value(spec: tuple, rng: random.Random = None) -> Any:
    """Sample a value according to the specification.
    
    Args:
        spec: Tuple of (type, *params)
        rng: Random number generator
        
    Returns:
        Sampled value
    """
    if rng is None:
        rng = random.Random()
        
    param_type = spec[0]
    
    if param_type == 'categorical':
        return rng.choice(spec[1])
    elif param_type == 'uniform':
        return rng.uniform(spec[1], spec[2])
    elif param_type == 'loguniform':
        log_min, log_max = np.log(spec[1]), np.log(spec[2])
        return np.exp(rng.uniform(log_min, log_max))
    elif param_type == 'int_uniform':
        return rng.randint(spec[1], spec[2])
    else:
        raise ValueError(f"Unknown parameter type: {param_type}")


def generate_grid_configs(search_space: dict) -> list[dict]:
    """Generate all combinations for grid search.
    
    Args:
        search_space: Dictionary mapping parameter paths to specs
        
    Returns:
        List of hyperparameter dictionaries
    """
    # Convert continuous params to discrete for grid search
    grid_values = {}
    for key, spec in search_space.items():
        param_type = spec[0]
        if param_type == 'categorical':
            grid_values[key] = spec[1]
        elif param_type in ('uniform', 'loguniform'):
            # Sample 3 points for continuous parameters
            if param_type == 'loguniform':
                grid_values[key] = list(np.exp(np.linspace(
                    np.log(spec[1]), np.log(spec[2]), 3
                )))
            else:
                grid_values[key] = list(np.linspace(spec[1], spec[2], 3))
        elif param_type == 'int_uniform':
            grid_values[key] = list(range(spec[1], spec[2] + 1, max(1, (spec[2] - spec[1]) // 3)))
    
    # Generate all combinations
    keys = list(grid_values.keys())
    values = [grid_values[k] for k in keys]
    
    configs = []
    for combination in itertools.product(*values):
        config = {k: v for k, v in zip(keys, combination)}
        configs.append(config)
    
    return configs


def generate_random_configs(search_space: dict, n_trials: int, seed: int = 42) -> list[dict]:
    """Generate random configurations for random search.
    
    Args:
        search_space: Dictionary mapping parameter paths to specs
        n_trials: Number of configurations to generate
        seed: Random seed
        
    Returns:
        List of hyperparameter dictionaries
    """
    rng = random.Random(seed)
    configs = []
    
    for _ in range(n_trials):
        config = {}
        for key, spec in search_space.items():
            config[key] = sample_value(spec, rng)
        configs.append(config)
    
    return configs


def apply_hyperparams(base_config: dict, hyperparams: dict) -> dict:
    """Apply hyperparameters to a base configuration.
    
    Args:
        base_config: Base configuration dictionary
        hyperparams: Dictionary of hyperparameter overrides
        
    Returns:
        New configuration with hyperparameters applied
    """
    config = copy.deepcopy(base_config)
    
    for key_path, value in hyperparams.items():
        # Handle special mapping from search space to config
        if key_path.startswith('ppo_params.') or key_path.startswith('network_params.'):
            # These go under learning_params
            actual_path = f"learning_params.{key_path}"
        else:
            actual_path = key_path
            
        set_nested_value(config, actual_path, value)
    
    return config


# ============================================================================
# Training Runner
# ============================================================================

def run_single_trial(
    config: dict,
    trial_id: int,
    output_base: Path,
    reduced_timesteps: bool = True,
    reduced_timesteps_factor: float = 0.1,
) -> dict:
    """Run a single training trial.
    
    Args:
        config: Full configuration dictionary
        trial_id: Trial identifier
        output_base: Base output directory
        reduced_timesteps: Whether to reduce training time for sweep
        reduced_timesteps_factor: Factor to reduce timesteps by
        
    Returns:
        Dictionary with trial results
    """
    import jax
    
    trial_dir = output_base / f"trial_{trial_id:04d}"
    os.makedirs(trial_dir, exist_ok=True)
    
    # Optionally reduce training time for hyperparameter search
    if reduced_timesteps:
        original_timesteps = config['learning_params']['ppo_params']['num_timesteps']
        config['learning_params']['ppo_params']['num_timesteps'] = int(
            original_timesteps * reduced_timesteps_factor
        )
        # Also reduce number of evaluations proportionally
        original_evals = config['learning_params']['ppo_params']['num_evals']
        config['learning_params']['ppo_params']['num_evals'] = max(
            5, int(original_evals * reduced_timesteps_factor)
        )
    
    # Save trial config
    config_path = trial_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(dict(config), f)
    
    try:
        # Create environment
        env, env_cfg = create_environment(config, for_training=True)
        eval_env, _ = create_environment(config, for_training=False)
        ppo_params, network_params = setup_training(config['learning_params'])
        
        # Training data tracking
        x_data, y_data, y_dataerr = [0], [0], [0]
        times = [datetime.datetime.now()]
        
        # Run training
        make_inference_fn, params, metrics = train(
            config, trial_dir, env, eval_env, 
            ppo_params, network_params, 
            times, x_data, y_data, y_dataerr
        )
        
        # Get final metrics
        final_reward = y_data[-1] if len(y_data) > 1 else 0.0
        max_reward = max(y_data) if len(y_data) > 1 else 0.0
        training_time = (times[-1] - times[0]).total_seconds()
        
        result = {
            'trial_id': trial_id,
            'status': 'completed',
            'final_reward': float(final_reward),
            'max_reward': float(max_reward),
            'training_time': training_time,
            'num_epochs': len(y_data) - 1,
            'reward_history': y_data,
        }
        
    except Exception as e:
        import traceback
        result = {
            'trial_id': trial_id,
            'status': 'failed',
            'error': str(e),
            'traceback': traceback.format_exc(),
            'final_reward': float('-inf'),
            'max_reward': float('-inf'),
        }
        
        # Save error info
        with open(trial_dir / 'error.txt', 'w') as f:
            f.write(result['traceback'])
    
    # Save result
    with open(trial_dir / 'result.json', 'w') as f:
        # Convert non-serializable items
        result_save = {k: v for k, v in result.items() if k != 'reward_history'}
        result_save['reward_history'] = [float(x) for x in result.get('reward_history', [])]
        json.dump(result_save, f, indent=2)
    
    return result


# ============================================================================
# Sweep Methods
# ============================================================================

def run_grid_search(
    base_config: dict,
    search_space: dict,
    output_dir: Path,
    **kwargs
) -> pd.DataFrame:
    """Run grid search over hyperparameters."""
    configs = generate_grid_configs(search_space)
    print(f"Grid search: {len(configs)} configurations")
    return _run_sweep(base_config, configs, output_dir, **kwargs)


def run_random_search(
    base_config: dict,
    search_space: dict,
    output_dir: Path,
    n_trials: int = 50,
    seed: int = 42,
    **kwargs
) -> pd.DataFrame:
    """Run random search over hyperparameters."""
    configs = generate_random_configs(search_space, n_trials, seed)
    print(f"Random search: {n_trials} trials")
    return _run_sweep(base_config, configs, output_dir, **kwargs)


def run_optuna_search(
    base_config: dict,
    search_space: dict,
    output_dir: Path,
    n_trials: int = 100,
    **kwargs
) -> pd.DataFrame:
    """Run Optuna-based hyperparameter optimization."""
    try:
        import optuna
    except ImportError:
        raise ImportError("Optuna not installed. Run: pip install optuna")
    
    results = []
    
    def objective(trial: optuna.Trial) -> float:
        # Sample hyperparameters using Optuna
        hyperparams = {}
        for key, spec in search_space.items():
            param_type = spec[0]
            name = key.replace('.', '_')  # Optuna doesn't like dots
            
            if param_type == 'categorical':
                # Handle list values (convert to string index)
                choices = spec[1]
                if isinstance(choices[0], list):
                    idx = trial.suggest_categorical(name, list(range(len(choices))))
                    hyperparams[key] = choices[idx]
                else:
                    hyperparams[key] = trial.suggest_categorical(name, choices)
            elif param_type == 'uniform':
                hyperparams[key] = trial.suggest_float(name, spec[1], spec[2])
            elif param_type == 'loguniform':
                hyperparams[key] = trial.suggest_float(name, spec[1], spec[2], log=True)
            elif param_type == 'int_uniform':
                hyperparams[key] = trial.suggest_int(name, spec[1], spec[2])
        
        # Apply to config and run trial
        config = apply_hyperparams(base_config, hyperparams)
        result = run_single_trial(config, trial.number, output_dir, **kwargs)
        
        # Store full result
        result['hyperparams'] = hyperparams
        results.append(result)
        
        return result['max_reward']
    
    # Create and run study
    study = optuna.create_study(
        direction='maximize',
        study_name='rl_hypersweep',
        storage=f'sqlite:///{output_dir}/optuna.db',
        load_if_exists=True,
    )
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Save best parameters
    best_params_path = output_dir / 'best_params.yaml'
    with open(best_params_path, 'w') as f:
        yaml.dump(study.best_params, f)
    
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best reward: {study.best_value:.2f}")
    print(f"Best params saved to: {best_params_path}")
    
    return _results_to_dataframe(results, search_space)


def _run_sweep(
    base_config: dict,
    configs: list[dict],
    output_dir: Path,
    **kwargs
) -> pd.DataFrame:
    """Run sweep over list of configurations."""
    results = []
    
    for i, hyperparams in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Trial {i+1}/{len(configs)}")
        print(f"Hyperparameters: {hyperparams}")
        print(f"{'='*60}\n")
        
        config = apply_hyperparams(base_config, hyperparams)
        result = run_single_trial(config, i, output_dir, **kwargs)
        result['hyperparams'] = hyperparams
        results.append(result)
        
        # Save intermediate results
        df = _results_to_dataframe(results, configs[0].keys())
        df.to_csv(output_dir / 'results.csv', index=False)
    
    return _results_to_dataframe(results, configs[0].keys())


def _results_to_dataframe(results: list[dict], param_keys) -> pd.DataFrame:
    """Convert results list to DataFrame."""
    rows = []
    for r in results:
        row = {
            'trial_id': r['trial_id'],
            'status': r['status'],
            'final_reward': r.get('final_reward', float('nan')),
            'max_reward': r.get('max_reward', float('nan')),
            'training_time': r.get('training_time', float('nan')),
        }
        # Add hyperparameters
        if 'hyperparams' in r:
            for key in param_keys:
                value = r['hyperparams'].get(key, None)
                # Convert lists to strings for CSV compatibility
                if isinstance(value, list):
                    value = str(value)
                row[key] = value
        rows.append(row)
    
    return pd.DataFrame(rows)


# ============================================================================
# Analysis Functions
# ============================================================================

def analyze_results(results_path: Path) -> None:
    """Analyze sweep results and print summary."""
    df = pd.read_csv(results_path)
    
    print("\n" + "="*60)
    print("HYPERPARAMETER SWEEP RESULTS")
    print("="*60)
    
    # Basic statistics
    completed = df[df['status'] == 'completed']
    print(f"\nTrials completed: {len(completed)}/{len(df)}")
    
    if len(completed) > 0:
        print(f"\nReward Statistics:")
        print(f"  Mean final reward: {completed['final_reward'].mean():.2f}")
        print(f"  Std final reward:  {completed['final_reward'].std():.2f}")
        print(f"  Max final reward:  {completed['final_reward'].max():.2f}")
        print(f"  Max reward (any):  {completed['max_reward'].max():.2f}")
        
        # Best trial
        best_idx = completed['max_reward'].idxmax()
        best_trial = completed.loc[best_idx]
        
        print(f"\nBest Trial (ID: {best_trial['trial_id']}):")
        print(f"  Max reward: {best_trial['max_reward']:.2f}")
        print(f"  Final reward: {best_trial['final_reward']:.2f}")
        
        # Print best hyperparameters
        print(f"\nBest Hyperparameters:")
        for col in df.columns:
            if col not in ['trial_id', 'status', 'final_reward', 'max_reward', 'training_time']:
                print(f"  {col}: {best_trial[col]}")
    
    print("\n" + "="*60)


def plot_sweep_results(results_path: Path, output_path: Path = None) -> None:
    """Generate visualization of sweep results."""
    import matplotlib.pyplot as plt
    
    df = pd.read_csv(results_path)
    completed = df[df['status'] == 'completed']
    
    if len(completed) == 0:
        print("No completed trials to plot")
        return
    
    # Get hyperparameter columns
    hp_cols = [c for c in df.columns if c not in 
               ['trial_id', 'status', 'final_reward', 'max_reward', 'training_time']]
    
    n_params = len(hp_cols)
    if n_params == 0:
        return
    
    # Create subplot grid
    n_cols = min(3, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_params == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, col in enumerate(hp_cols):
        ax = axes[i]
        
        # Check if numeric
        try:
            x = pd.to_numeric(completed[col])
            ax.scatter(x, completed['max_reward'], alpha=0.6)
            ax.set_xlabel(col.split('.')[-1])
        except (ValueError, TypeError):
            # Categorical - use box plot
            categories = completed[col].unique()
            data = [completed[completed[col] == c]['max_reward'] for c in categories]
            ax.boxplot(data, labels=[str(c)[:20] for c in categories])
            ax.set_xlabel(col.split('.')[-1])
            ax.tick_params(axis='x', rotation=45)
        
        ax.set_ylabel('Max Reward')
        ax.set_title(col.split('.')[-1])
    
    # Hide empty subplots
    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Hyperparameter sweep for RL training')
    parser.add_argument('config', type=str, help='Path to base config YAML file')
    parser.add_argument('--method', type=str, default='random',
                        choices=['grid', 'random', 'optuna'],
                        help='Search method (default: random)')
    parser.add_argument('--n-trials', type=int, default=20,
                        help='Number of trials for random/optuna search')
    parser.add_argument('--search-space', type=str, default='minimal',
                        choices=['full', 'minimal'],
                        help='Search space to use (default: minimal)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: results/sweeps/<timestamp>)')
    parser.add_argument('--reduced-timesteps', action='store_true', default=True,
                        help='Use reduced training time for faster sweeps')
    parser.add_argument('--timesteps-factor', type=float, default=0.1,
                        help='Factor to reduce timesteps by (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--analyze', type=str, default=None,
                        help='Path to results.csv to analyze (skip training)')
    
    args = parser.parse_args()
    
    # Analysis mode
    if args.analyze:
        analyze_results(Path(args.analyze))
        plot_path = Path(args.analyze).parent / 'sweep_analysis.png'
        plot_sweep_results(Path(args.analyze), plot_path)
        return
    
    # Setup
    run_setup()
    
    # Load base config
    with open(args.config, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Select search space
    search_space = SEARCH_SPACE_MINIMAL if args.search_space == 'minimal' else SEARCH_SPACE
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    robot = base_config.get('robot', 'robot')
    env = base_config.get('env', 'env')
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path('results/sweeps') / f'{robot}_{env}_{timestamp}'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save sweep configuration
    sweep_config = {
        'base_config_path': args.config,
        'method': args.method,
        'n_trials': args.n_trials,
        'search_space': args.search_space,
        'reduced_timesteps': args.reduced_timesteps,
        'timesteps_factor': args.timesteps_factor,
        'seed': args.seed,
        'git_hash': get_commit_hash(),
    }
    with open(output_dir / 'sweep_config.yaml', 'w') as f:
        yaml.dump(sweep_config, f)
    
    print(f"\n{'='*60}")
    print(f"HYPERPARAMETER SWEEP")
    print(f"{'='*60}")
    print(f"Method: {args.method}")
    print(f"Search space: {args.search_space} ({len(search_space)} parameters)")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    # Run sweep
    kwargs = {
        'reduced_timesteps': args.reduced_timesteps,
        'reduced_timesteps_factor': args.timesteps_factor,
    }
    
    if args.method == 'grid':
        results_df = run_grid_search(base_config, search_space, output_dir, **kwargs)
    elif args.method == 'random':
        results_df = run_random_search(
            base_config, search_space, output_dir,
            n_trials=args.n_trials, seed=args.seed, **kwargs
        )
    elif args.method == 'optuna':
        results_df = run_optuna_search(
            base_config, search_space, output_dir,
            n_trials=args.n_trials, **kwargs
        )
    
    # Save final results
    results_path = output_dir / 'results.csv'
    results_df.to_csv(results_path, index=False)
    
    # Analyze and plot
    analyze_results(results_path)
    plot_sweep_results(results_path, output_dir / 'sweep_analysis.png')
    
    print(f"\nSweep complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
