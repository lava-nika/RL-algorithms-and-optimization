import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import gymnasium as gym
from algorithms.sarsa import SemiGradientNStepSarsa
from algorithms.reinforce import REINFORCE
from algorithms.actor_critic import ActorCritic
import itertools
from datetime import datetime
import json

def evaluate_with_stats(algo_class, params, n_episodes=1000, n_runs=3, algo_name=""):
    all_rewards = []
    
    print(f"   Running {n_runs} trials...", end='', flush=True)
    
    for run in range(n_runs):
        algo = algo_class(**params)
        rewards = []
        
        for episode in range(n_episodes):
            env = gym.make('CartPole-v1')
            reward = algo.train_episode(env)
            rewards.append(reward)
            env.close()
        
        all_rewards.append(rewards)
        print(f" {run+1}", end='', flush=True)
    
    print(" done.")
    
    all_rewards = np.array(all_rewards)
    
    final_rewards = np.mean(all_rewards[:, -100:], axis=1)
    mean_final = np.mean(final_rewards)
    std_final = np.std(final_rewards)
    best_final = np.max(final_rewards)
    
    return {
        'mean': mean_final,
        'std': std_final,
        'best': best_final
    }


def tune_sarsa():
    print("1. Semi-gradient n-step SARSA tuning")

    n_steps_options = [7, 8, 9]
    alpha_options = [0.18, 0.20, 0.22]
    epsilon_options = [0.04, 0.05, 0.06]
    gamma = 0.99
    
    total = len(n_steps_options) * len(alpha_options) * len(epsilon_options)
    print(f"\nTesting {total} configurations * 3 runs = {total*3} training runs")
    
    results = []
    count = 0
    
    for n_steps, alpha, epsilon in itertools.product(n_steps_options, alpha_options, epsilon_options):
        count += 1
        params = {'n_steps': n_steps, 'alpha': alpha, 'gamma': gamma, 'epsilon': epsilon}
        
        print(f"[{count}/{total}] n={n_steps}, alpha={alpha}, epsilon={epsilon}")
        
        stats = evaluate_with_stats(SemiGradientNStepSarsa, params, n_episodes=1000, n_runs=3)
        
        print(f"   -> {stats['mean']:.2f} ± {stats['std']:.2f} (best: {stats['best']:.2f})")
        
        if stats['best'] >= 475:
            print(f"   SOLVED, Best run achieved 475+")
        
        results.append({'params': params, **stats})
    
    results.sort(key=lambda x: x['mean'], reverse=True)
    
    print("Top 5 Semi-gradient n-step SARSA configs:")
    for i, r in enumerate(results[:5], 1):
        p = r['params']
        solved = "SOLVED" if r['best'] >= 475 else f"({r['mean']/475*100:.1f}%)"
        print(f"{i}. n={p['n_steps']}, alpha={p['alpha']}, epsilon={p['epsilon']}: "
              f"{r['mean']:.2f}±{r['std']:.2f} (best:{r['best']:.2f}) {solved}")
    
    return results


def tune_reinforce():
    print("2. REINFORCE with baseline tuning")
    
    alpha_theta_options = [0.0006, 0.0008, 0.0010]
    alpha_w_options = [0.006, 0.008, 0.010]
    gamma_options = [0.98, 0.99, 0.995]
    
    total = len(alpha_theta_options) * len(alpha_w_options) * len(gamma_options)
    print(f"\nTesting {total} configurations * 3 runs = {total*3} training runs")
    
    results = []
    count = 0
    
    for alpha_theta, alpha_w, gamma in itertools.product(alpha_theta_options, alpha_w_options, gamma_options):
        count += 1
        params = {'alpha_theta': alpha_theta, 'alpha_w': alpha_w, 'gamma': gamma}
        
        print(f"[{count}/{total}] alpha_theta={alpha_theta}, alpha_w={alpha_w}, gamma={gamma}")
        
        stats = evaluate_with_stats(REINFORCE, params, n_episodes=1000, n_runs=3)
        
        print(f"   -> {stats['mean']:.2f} ± {stats['std']:.2f} (best: {stats['best']:.2f})")
        
        results.append({'params': params, **stats})
    
    results.sort(key=lambda x: x['mean'], reverse=True)
    
    print("Top 5 REINFORCE configs:")
    for i, r in enumerate(results[:5], 1):
        p = r['params']
        print(f"{i}. alpha_theta={p['alpha_theta']}, alpha_w={p['alpha_w']}, gamma={p['gamma']}: "
              f"{r['mean']:.2f}±{r['std']:.2f} (best:{r['best']:.2f})")
    
    return results


def tune_actor_critic():
    print("3. One-step Actor-critic tuning")
    
    alpha_theta_options = [0.0003, 0.0005, 0.0008, 0.0010, 0.0012, 0.0015]
    alpha_w_options = [0.003, 0.005, 0.008, 0.010, 0.012, 0.015]
    gamma_options = [0.98, 0.99, 0.995, 0.999]
    
    configs_to_test = []
    
    for alpha_theta in alpha_theta_options:
        for alpha_w in alpha_w_options:
            for gamma in gamma_options:
                configs_to_test.append({
                    'alpha_theta': alpha_theta,
                    'alpha_w': alpha_w,
                    'gamma': gamma
                })
    
    special_configs = [
        {'alpha_theta': 0.0005, 'alpha_w': 0.0005, 'gamma': 0.99},
        {'alpha_theta': 0.0008, 'alpha_w': 0.0008, 'gamma': 0.99},
        {'alpha_theta': 0.001, 'alpha_w': 0.001, 'gamma': 0.99},
    
        {'alpha_theta': 0.0001, 'alpha_w': 0.001, 'gamma': 0.995},
        {'alpha_theta': 0.0002, 'alpha_w': 0.002, 'gamma': 0.995},

        {'alpha_theta': 0.0003, 'alpha_w': 0.003, 'gamma': 0.999},
        {'alpha_theta': 0.0005, 'alpha_w': 0.005, 'gamma': 0.999},
    ]
    
    configs_to_test.extend(special_configs)
    
    unique_configs = []
    seen = set()
    for config in configs_to_test:
        key = (config['alpha_theta'], config['alpha_w'], config['gamma'])
        if key not in seen:
            seen.add(key)
            unique_configs.append(config)
    
    total = len(unique_configs)
    print(f"\nTesting {total} configurations * 3 runs = {total*3} training runs")
    
    results = []
    count = 0
    best_so_far = 34.33
    
    for params in unique_configs:
        count += 1
        print(f"[{count}/{total}] alpha_theta={params['alpha_theta']}, alpha_w={params['alpha_w']}, gamma={params['gamma']}")
        
        stats = evaluate_with_stats(ActorCritic, params, n_episodes=1000, n_runs=3)
        
        print(f"   -> {stats['mean']:.2f} ± {stats['std']:.2f} (best: {stats['best']:.2f})")
        
        if stats['mean'] > best_so_far:
            improvement = ((stats['mean'] - best_so_far) / best_so_far) * 100
            print(f"   NEW BEST +{improvement:.1f}% improvement")
            best_so_far = stats['mean']
        
        results.append({'params': params, **stats})
    
    results.sort(key=lambda x: x['mean'], reverse=True)
    
    print("Top 10 One-step actor-critic configs:")
    for i, r in enumerate(results[:10], 1):
        p = r['params']
        print(f"{i:2d}. alpha_theta={p['alpha_theta']:.4f}, alpha_w={p['alpha_w']:.4f}, gamma={p['gamma']}: "
              f"{r['mean']:6.2f}±{r['std']:5.2f} (best:{r['best']:6.2f})")
    
    return results


def generate_json_results(sarsa_results, reinforce_results, ac_results):
    best_sarsa = sarsa_results[0]
    best_reinforce = reinforce_results[0]
    best_ac = ac_results[0]
    
    print("\n1. Semi-gradient n-step SARSA:")
    print(f"   Performance: {best_sarsa['mean']:.2f} ± {best_sarsa['std']:.2f}")
    print(f"   Best run: {best_sarsa['best']:.2f}")
    if best_sarsa['best'] >= 475:
        print(f"   Status: SOLVED")
    else:
        print(f"   Status: {best_sarsa['mean']/475*100:.1f}% to solved")
    print(f"   Parameters: {best_sarsa['params']}")
    
    print("\n2. REINFORCE with baseline:")
    print(f"   Performance: {best_reinforce['mean']:.2f} ± {best_reinforce['std']:.2f}")
    print(f"   Best run: {best_reinforce['best']:.2f}")
    print(f"   Status: {best_reinforce['mean']/475*100:.1f}% to solved")
    print(f"   Parameters: {best_reinforce['params']}")
    
    print("\n3. One-step Actor-critic:")
    print(f"   Performance: {best_ac['mean']:.2f} ± {best_ac['std']:.2f}")
    print(f"   Best run: {best_ac['best']:.2f}")
    print(f"   Status: {best_ac['mean']/475*100:.1f}% to solved")
    print(f"   Parameters: {best_ac['params']}")
    
    results_data = {
        'sarsa': {'best': best_sarsa['params'], 'all': [{'params': r['params'], 'mean': float(r['mean']), 'std': float(r['std'])} for r in sarsa_results[:10]]},
        'reinforce': {'best': best_reinforce['params'], 'all': [{'params': r['params'], 'mean': float(r['mean']), 'std': float(r['std'])} for r in reinforce_results[:10]]},
        'actor_critic': {'best': best_ac['params'], 'all': [{'params': r['params'], 'mean': float(r['mean']), 'std': float(r['std'])} for r in ac_results[:10]]}
    }
    
    output_dir = Path(__file__).parent
    results_file = output_dir / 'tuning_results.json'
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to '{results_file}'")
    


if __name__ == "__main__":
    start_time = datetime.now()
    print(f"\nStart time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    input("Press enter to start full hyperparameter tuning")
    
    try:
        sarsa_results = tune_sarsa()
        reinforce_results = tune_reinforce()
        ac_results = tune_actor_critic()
        
        generate_json_results(sarsa_results, reinforce_results, ac_results)
        
    except KeyboardInterrupt:
        print("\n\nTuning interrupted by user")
    
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"Total time: {duration}")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nTuning complete")
