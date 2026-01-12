import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import gymnasium as gym
import json
import time
from datetime import datetime
from algorithms.pi2_cmaes_tiles import PI2_CMAES_Tiles


def evaluate_config(config, n_generations=100, n_runs=3, verbose=False):
    env = gym.make('CartPole-v1')
    
    run_rewards = []
    
    for run in range(n_runs):
        if verbose:
            print(f"    Run {run+1}/{n_runs}...", end=" ")
        
        agent = PI2_CMAES_Tiles(
            n_actions=env.action_space.n,
            population_size=config['population_size'],
            lambda_=config['lambda'],
            elite_ratio=config['elite_ratio'],
            sigma_init=config['sigma_init'],
            num_tilings=config['num_tilings'],
            tiles_per_dim=config['tiles_per_dim']
        )
        
        agent.train(env, n_generations=n_generations, verbose=False)
        
        mean_reward, std_reward = agent.evaluate(env, n_episodes=100)
        run_rewards.append(mean_reward)
        
        if verbose:
            print(f"{mean_reward:.1f}")
    
    env.close()
    
    mean = np.mean(run_rewards)
    std = np.std(run_rewards)
    
    return {
        'config': config,
        'mean_reward': mean,
        'std_reward': std,
        'run_rewards': run_rewards
    }


def full_tuning():
    print("Hyperparameter tuning for PI2-CMA-ES (with tile coding)")
    configs = []
    
    baseline_quick = {
        'population_size': 15,
        'lambda': 10.0,
        'elite_ratio': 0.3,
        'sigma_init': 1.0,
        'num_tilings': 8,
        'tiles_per_dim': 4
    }
    configs.append(('Best from quick tuning', baseline_quick.copy()))
    
    for sigma in [0.8, 1.2, 1.5]:
        config = baseline_quick.copy()
        config['sigma_init'] = sigma
        configs.append((f'Sigma={sigma}', config))
    
    for pop_size in [20, 25, 30]:
        config = baseline_quick.copy()
        config['population_size'] = pop_size
        configs.append((f'Pop={pop_size}, sigma=1.0', config))

    for lambda_val in [15.0, 20.0, 30.0]:
        config = baseline_quick.copy()
        config['lambda'] = lambda_val
        configs.append((f'Lambda={lambda_val}, sigma=1.0', config))
    
    for elite in [0.2, 0.4, 0.5]:
        config = baseline_quick.copy()
        config['elite_ratio'] = elite
        configs.append((f'Elite={elite}, sigma=1.0', config))
    
    for num_tilings in [10, 12, 14, 16]:
        config = baseline_quick.copy()
        config['num_tilings'] = num_tilings
        configs.append((f'{num_tilings} tilings, sigma=1.0', config))
    
    for tiles in [5, 6]:
        config = baseline_quick.copy()
        config['tiles_per_dim'] = tiles
        configs.append((f'{tiles} tiles/dim, sigma=1.0', config))
    
    config = baseline_quick.copy()
    config['population_size'] = 25
    config['num_tilings'] = 12
    configs.append(('Combined 1: Pop25+T12', config))
    
    config = baseline_quick.copy()
    config['lambda'] = 20.0
    config['num_tilings'] = 12
    configs.append(('Combined 2: Lambda20+T12', config))
    
    config = baseline_quick.copy()
    config['population_size'] = 25
    config['lambda'] = 20.0
    configs.append(('Combined 3: Pop25+Lambda20', config))
    
    config = baseline_quick.copy()
    config['sigma_init'] = 1.5
    config['elite_ratio'] = 0.4
    config['lambda'] = 20.0
    configs.append(('Aggressive explore', config))
    
    config = baseline_quick.copy()
    config['population_size'] = 30
    config['num_tilings'] = 14
    configs.append(('Combined 4: Pop30+T14', config))
    
    config = baseline_quick.copy()
    config['population_size'] = 20
    config['lambda'] = 15.0
    config['num_tilings'] = 10
    config['elite_ratio'] = 0.35
    configs.append(('Balanced optimal', config))
    
    config = {
        'population_size': 30,
        'lambda': 25.0,
        'elite_ratio': 0.5,
        'sigma_init': 1.5,
        'num_tilings': 10,
        'tiles_per_dim': 5
    }
    configs.append(('Very high explore', config))
    
    config = baseline_quick.copy()
    config['num_tilings'] = 16
    config['tiles_per_dim'] = 5
    configs.append(('Dense features', config))
    
    print(f"Total configurations to test = {len(configs)}")
    print(f"Generations per config = 100")
    print(f"Runs per config = 3")
    print(f"Total training runs = {len(configs) * 3}")
    print()
    
    results = []
    start_time = time.time()
    
    for i, (name, config) in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] {name}:")
        print(f"  Config: Pop={config['population_size']}, "
              f"Lambda={config['lambda']}, Elite={config['elite_ratio']}")
        print(f"          sigma={config['sigma_init']}, "
              f"Tiles={config['num_tilings']}*{config['tiles_per_dim']}")
        
        result = evaluate_config(config, n_generations=100, n_runs=3, verbose=True)
        result['name'] = name
        results.append(result)
        
        mean = result['mean_reward']
        std = result['std_reward']
        percent = mean / 475 * 100
        
        print(f"  -> Mean: {mean:.2f} ± {std:.2f} ({percent:.1f}% to solved)")
        
        elapsed = time.time() - start_time
        avg_time = elapsed / i
        remaining = avg_time * (len(configs) - i)
        print(f"  Time: {elapsed/60:.1f}m elapsed, {remaining/60:.1f}m remaining")
    

    results.sort(key=lambda x: x['mean_reward'], reverse=True)
    
    output_dir = Path(__file__).parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_dir / f'pi2_tiles_full_tuning_{timestamp}.json'
    
    results_json = []
    for result in results:
        results_json.append({
            'name': result['name'],
            'config': result['config'],
            'mean_reward': float(result['mean_reward']),
            'std_reward': float(result['std_reward']),
            'run_rewards': [float(r) for r in result['run_rewards']]
        })
    
    with open(filename, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print("Full tuning complete")
    print(f"\nResults saved to: {filename}")
    print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")
    
    print("Top 10 configs:")
    
    for i, result in enumerate(results[:10], 1):
        config = result['config']
        mean = result['mean_reward']
        std = result['std_reward']
        percent = mean / 475 * 100
          
        print(f"\n{result['name']}")
        print(f"   Reward: {mean:.2f} ± {std:.2f} ({percent:.1f}% to solved)")
        print(f"   Pop={config['population_size']}, lambda={config['lambda']}, "
              f"Elite={config['elite_ratio']}, sigma={config['sigma_init']}")
        print(f"   Tiles={config['num_tilings']}*{config['tiles_per_dim']}")
    
    
    print("\nJSON results saved")
    
    return results

if __name__ == '__main__':
    results = full_tuning()
