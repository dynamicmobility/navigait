import pandas as pd
import matplotlib.pyplot as plt
from utils.plotting import set_mpl_params
import yaml
from pathlib import Path

PATHS = {
    # 'navigait0':  'icra-policies/navigait/',
    # 'navigait1':  'icra2026-final/navigait/2e8_096discount/', #'icra2026-final/navigait/long2e8/progress.csv',# 'icra2026-final/navigait/long/progress.csv',
    # 'navigait2': 'icra2026-final/navigait/long2e8/',
    'navigait': 'icra2026-final/navigait/long/',
    'Canonical': 'icra2026-final/canonical/4e8/',
    'Imitation': 'icra2026-final/imitation/test'
}

def main():
    def open_progress(name):
        path = f'icra-policies/{name}/progress.csv'
        path = Path(PATHS[name]) / 'progress.csv'
        df = pd.read_csv(path)
        return df
    
    def get_max_reward(name):
        with open(Path(PATHS[name]) / 'config.yaml', 'r') as file:
            data = yaml.safe_load(file)
        sum = 0
        rewards = data['env_config']['reward']['weights']
        for weight in rewards:
            if float(rewards[weight]) > 0:
                sum += float(rewards[weight])
        
        return sum * data['learning_params']['ppo_params']['episode_length']

    to_plot = ['navigait', 'Imitation', 'Canonical']
    dfs = [open_progress(name) for name in to_plot]
    maxes = [get_max_reward(name) for name in to_plot]

    set_mpl_params()
    fig, ax = plt.subplots()

    def plot_progress(df, color, label, max):
        timesteps = df['x']
        avg_normalize_return = df['y'] / max
        normalized_std = df['yerr'] / max
        ax.plot(timesteps, avg_normalize_return, label=label, color=color)
        ax.scatter(timesteps, avg_normalize_return, color=color, s=8)
        ax.fill_between(timesteps, avg_normalize_return - normalized_std, avg_normalize_return + normalized_std, color=color, alpha=0.3)  # shaded band
        return ax
    
    FONTSIZE = 20
    colors = ['red', 'green', 'blue', 'yellow', 'orange', 'pink']
    labels = to_plot
    for df, max_val, label, color in zip(dfs, maxes, labels, colors):
        if 'navigait' in label.lower():
            label = label.replace('navigait', r'\textsc{NaviGait}').replace('Navigait', r'\textsc{NaviGait}').replace('NAVIGAIT', r'\textsc{NaviGait}')
        ax = plot_progress(df, color, label, max_val)

    ax.set_xlabel('Iterations', fontsize=FONTSIZE)
    ax.set_ylabel('Return', fontsize=FONTSIZE)
    ax.legend(fontsize=FONTSIZE)
    ax.tick_params(labelsize=FONTSIZE)
    ax.grid(True)
    fig.set_size_inches((13, 5))
    fig.tight_layout()
    fig.savefig('paper_plots/training_times.svg', dpi=400)

    

if __name__ == '__main__':
    main()