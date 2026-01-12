import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from algorithms.sarsa import SemiGradientNStepSarsa
from algorithms.reinforce import REINFORCE
from algorithms.actor_critic import ActorCritic
from algorithms.pi2_cmaes_tiles import PI2_CMAES_Tiles


def moving_average(data, window=10):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')


def train_all_algorithms(n_episodes=500, n_generations=100):
    print("CartPole-v1 RL algorithms training with optimal hyperparameters")
    
    results = {}
    
    episode_algorithms = {
        'SARSA': {
            'algo': SemiGradientNStepSarsa(n_steps=9, alpha=0.22, gamma=0.99, epsilon=0.06),
            'color': '#1f77b4', 'linestyle': '-', 'linewidth': 2.5
        },
        'REINFORCE': {
            'algo': REINFORCE(alpha_theta=0.0006, alpha_w=0.01, gamma=0.99),
            'color': '#ff7f0e', 'linestyle': '-', 'linewidth': 2.5
        },
        'Actor-Critic': {
            'algo': ActorCritic(alpha_theta=0.0015, alpha_w=0.01, gamma=0.999),
            'color': '#2ca02c', 'linestyle': '-', 'linewidth': 2.0
        }
    }
    
    print(f"\nTraining episode-based algorithms ({n_episodes} episodes)")
    
    for name, config in episode_algorithms.items():
        print(f"Training {name}...", end=' ', flush=True)
        algo = config['algo']
        rewards = []
        
        for episode in range(n_episodes):
            env = gym.make('CartPole-v1')
            reward = algo.train_episode(env)
            rewards.append(reward)
            env.close()
            
            if (episode + 1) % 100 == 0:
                print(f"{episode+1}", end='...', flush=True)
        
        final_avg = np.mean(rewards[-100:])
        print(f" Done, final 100-ep avg: {final_avg:.2f}")
        
        results[name] = {
            'rewards': rewards, 'x_axis': list(range(1, len(rewards) + 1)),
            'type': 'episode', 'color': config['color'],
            'linestyle': config['linestyle'], 'linewidth': config['linewidth']
        }
    
    print(f"\nTraining population-based algorithm ({n_generations} generations)")
    print("Training PI2-CMA-ES (Tiles, Tuned)...", end=' ', flush=True)
    
    pi2_algo = PI2_CMAES_Tiles(
        population_size=15, lambda_=30.0, sigma_init=1.0, elite_ratio=0.3,
        num_tilings=8, tiles_per_dim=4
    )
    
    env = gym.make('CartPole-v1')
    best_rewards = pi2_algo.train(env, n_generations=n_generations, verbose=False)
    env.close()
    
    mean_rewards = [reward[0] if isinstance(reward, tuple) else reward for reward in best_rewards]
    
    episode_equivalents = [(i+1) * 15 for i in range(len(mean_rewards))]
    final_avg = np.mean(mean_rewards[-min(20, len(mean_rewards)):])
    print(f" Done! Final avg: {final_avg:.2f}")
    
    results['PI2-CMA-ES (Tuned)'] = {
        'rewards': mean_rewards, 'x_axis': episode_equivalents, 'type': 'generation',
        'color': '#e377c2', 'linestyle': '-', 'linewidth': 3.0
    }
    
    return results


def create_plots(results, output_file='cartpole_final.png'):
    print("Creating learning curves")
    
    plt.rcParams.update({
        'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 18,
        'legend.fontsize': 14, 'xtick.labelsize': 14, 'ytick.labelsize': 14
    })
    
    fig1, ax1 = plt.subplots(figsize=(16, 9))
    
    episode_based = ['Actor-Critic', 'SARSA', 'REINFORCE']
    
    for name in episode_based:
        if name not in results:
            continue
        
        data = results[name]
        x, y = data['x_axis'], data['rewards']
        
        if len(y) > 30:
            window = max(20, len(y) // 25)
            y_smooth = moving_average(y, window=window)
            x_smooth = x[window-1:] if len(x) > window else x[:len(y_smooth)]
            
            ax1.plot(x_smooth, y_smooth, label=name, color=data['color'],
                    linestyle=data['linestyle'], linewidth=2.5, alpha=0.9)
        else:
            ax1.plot(x, y, label=name, color=data['color'],
                    linestyle=data['linestyle'], linewidth=2.5, alpha=0.9)
    
    ax1.axhline(y=475, color='red', linestyle=':', linewidth=2.5,
               alpha=0.8, label='Solved (475)', zorder=1)
    
    ax1.set_xlabel('Episode', fontsize=16, fontweight='normal')
    ax1.set_ylabel('Return (Smoothed)', fontsize=16, fontweight='normal')
    ax1.set_title('CartPole-v1 Learning Curves: Episode-Based Algorithms', 
                  fontsize=18, fontweight='bold', pad=15)
    
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.set_ylim(bottom=0, top=500)
    ax1.set_xlim(left=0, right=600)
    
    handles1, labels1 = ax1.get_legend_handles_labels()
    by_label1 = dict(zip(labels1, handles1))
    ax1.legend(by_label1.values(), by_label1.keys(), loc='lower right', 
              framealpha=0.95, edgecolor='gray', fancybox=False, shadow=False)
    
    plt.tight_layout()
    episode_plot = output_file.replace('.png', '_episode_based.png')
    plt.savefig(episode_plot, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Episode-based plot saved: {episode_plot}")
    plt.close()
    
    if 'PI2-CMA-ES (Tuned)' in results:
        fig2, ax2 = plt.subplots(figsize=(16, 9))
        
        data = results['PI2-CMA-ES (Tuned)']
        x, y = data['x_axis'], data['rewards']
        
        if len(y) > 30:
            window = max(20, len(y) // 25)
            y_smooth = moving_average(y, window=window)
            x_smooth = x[window-1:] if len(x) > window else x[:len(y_smooth)]
            
            ax2.plot(x_smooth, y_smooth, label='PI²-CMA-ES (Tuned)', color=data['color'],
                    linestyle=data['linestyle'], linewidth=2.5, alpha=0.9)
        else:
            ax2.plot(x, y, label='PI²-CMA-ES (Tuned)', color=data['color'],
                    linestyle=data['linestyle'], linewidth=2.5, alpha=0.9)
        
        ax2.axhline(y=475, color='red', linestyle=':', linewidth=2.5,
                   alpha=0.8, label='Solved (475)', zorder=1)
        
        ax2.set_xlabel('Episode Equivalent', fontsize=16, fontweight='normal')
        ax2.set_ylabel('Return (Smoothed)', fontsize=16, fontweight='normal')
        ax2.set_title('CartPole-v1 Learning Curve: PI²-CMA-ES with Tile Coding', 
                      fontsize=18, fontweight='bold', pad=15)
        
        ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax2.set_ylim(bottom=0, top=500)
        ax2.set_xlim(left=0)
        
        handles2, labels2 = ax2.get_legend_handles_labels()
        by_label2 = dict(zip(labels2, handles2))
        ax2.legend(by_label2.values(), by_label2.keys(), loc='lower right', 
                  framealpha=0.95, edgecolor='gray', fancybox=False, shadow=False)
        
        plt.tight_layout()
        pi2_plot = output_file.replace('.png', '_pi2_cmaes.png')
        plt.savefig(pi2_plot, dpi=300, bbox_inches='tight', facecolor='white')
        print(f" PI2-CMA-ES plot saved: {pi2_plot}")
        plt.close()
    
    return fig1


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train and compare RL algorithms on CartPole-v1 with optimal hyperparameters'
    )
    parser.add_argument('--episodes', type=int, default=500,
                       help='Number of episodes for episode-based algorithms (default: 500)')
    parser.add_argument('--generations', type=int, default=100,
                       help='Number of generations for PI2-CMA-ES (default: 100)')
    args = parser.parse_args()
    
    results = train_all_algorithms(n_episodes=args.episodes, n_generations=args.generations)
    create_plots(results, output_file='cartpole_final.png')


if __name__ == "__main__":
    main()
