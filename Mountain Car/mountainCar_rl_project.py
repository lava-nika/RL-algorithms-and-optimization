"""
## Project Overview

This code implements and evaluates three reinforcement learning algorithms from scratch:
1. **One-Step Actor-Critic** (Section 13.5 of Sutton & Barto)
2. **Episodic Semi-Gradient n-step SARSA** (Section 10.2 of Sutton & Barto)
3. **REINFORCE with Baseline** (Section 13.4 of Sutton & Barto)

These algorithms are evaluated on the MountainCar-v0 environment.
"""

## 1. Import Libraries and Setup

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from collections import defaultdict
import seaborn as sns
from tqdm import tqdm
import pandas as pd

np.random.seed(42)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


## 2. Feature Representation:

class TileCoding:
    def __init__(self, num_tilings: int, tiles_per_dim: List[int],
                 state_bounds: List[Tuple[float, float]]):

        self.num_tilings = num_tilings
        self.tiles_per_dim = np.array(tiles_per_dim)
        self.state_bounds = np.array(state_bounds)
        self.num_dims = len(state_bounds)

        self.tile_widths = (self.state_bounds[:, 1] - self.state_bounds[:, 0]) / self.tiles_per_dim

        self.tiles_per_tiling = np.prod(self.tiles_per_dim)
        self.num_features = int(self.num_tilings * self.tiles_per_tiling)

        self.offsets = np.random.uniform(0, self.tile_widths,
                                         (self.num_tilings, self.num_dims))

    def get_features(self, state: np.ndarray) -> np.ndarray:
        features = np.zeros(self.num_features)
        state = np.clip(state, self.state_bounds[:, 0], self.state_bounds[:, 1])

        for tiling_idx in range(self.num_tilings):
            offset_state = state - self.offsets[tiling_idx]

            tile_indices = np.floor((offset_state - self.state_bounds[:, 0]) /
                                   self.tile_widths).astype(int)
            tile_indices = np.clip(tile_indices, 0, self.tiles_per_dim - 1)

            flat_idx = 0
            multiplier = 1
            for i in range(self.num_dims - 1, -1, -1):
                flat_idx += tile_indices[i] * multiplier
                multiplier *= self.tiles_per_dim[i]

            feature_idx = int(tiling_idx * self.tiles_per_tiling + flat_idx)
            features[feature_idx] = 1.0

        return features

    def get_num_features(self) -> int:
        return self.num_features


## 3. Algorithm Implementations

### 3.1 One-Step Actor-Critic

#### Pseudocode
"""
Algorithm: One-Step Actor-Critic (episodic)
Input: policy π(a|s,θ), value function V(s,w)
Parameters: step sizes αθ, αw

Initialize policy parameters θ and value function weights w
For each episode:
    Initialize S (first state of episode)
    I ← 1
    While S is not terminal:
        A ~ π(·|S, θ)
        Take action A, observe R, S'
        δ ← R + γ·V(S',w) - V(S,w)  (if S' is terminal, V(S',w)=0)
        w ← w + αw·δ·∇V(S,w)
        θ ← θ + αθ·I·δ·∇ln π(A|S,θ)
        I ← γI
        S ← S'
"""

class OneStepActorCritic:
    def __init__(self, feature_extractor: TileCoding, num_actions: int,
                 alpha_theta: float = 0.001, alpha_w: float = 0.01,
                 gamma: float = 0.99):
        self.feature_extractor = feature_extractor
        self.num_actions = num_actions
        self.alpha_theta = alpha_theta
        self.alpha_w = alpha_w
        self.gamma = gamma

        num_features = feature_extractor.get_num_features()
        self.w = np.zeros(num_features)
        self.theta = np.zeros((num_actions, num_features))

    def get_value(self, state: np.ndarray) -> float:
        features = self.feature_extractor.get_features(state)
        return np.dot(self.w, features)

    def get_action_preferences(self, state: np.ndarray) -> np.ndarray:
        features = self.feature_extractor.get_features(state)
        return self.theta @ features

    def get_policy(self, state: np.ndarray) -> np.ndarray:
        preferences = self.get_action_preferences(state)
        exp_prefs = np.exp(preferences - np.max(preferences))
        return exp_prefs / np.sum(exp_prefs)

    def select_action(self, state: np.ndarray) -> int:
        probs = self.get_policy(state)
        return np.random.choice(self.num_actions, p=probs)

    def train_episode(self, env) -> float:
        state, _ = env.reset()
        total_reward = 0
        I = 1.0

        done = False
        while not done:
            action = self.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            features = self.feature_extractor.get_features(state)

            v_current = np.dot(self.w, features)
            v_next = 0 if done else self.get_value(next_state)
            delta = reward + self.gamma * v_next - v_current

            self.w += self.alpha_w * delta * features

            probs = self.get_policy(state)
            grad_ln_pi = np.outer(np.eye(self.num_actions)[action] - probs, features)
            self.theta += self.alpha_theta * I * delta * grad_ln_pi

            I *= self.gamma
            state = next_state

        return total_reward


### 3.2 Episodic Semi-Gradient n-step SARSA

#### Pseudocode
"""
Algorithm: Episodic Semi-Gradient n-step SARSA
Input: step size α, n-step parameter n, ε for ε-greedy
Initialize Q(s,a,w) arbitrarily

For each episode:
    Initialize S0
    Select A0 ~ ε-greedy(S0)
    T ← ∞
    For t = 0, 1, 2, ... :
        If t < T:
            Take action At, observe Rt+1, St+1
            If St+1 is terminal:
                T ← t + 1
            Else:
                Select At+1 ~ ε-greedy(St+1)
        τ ← t - n + 1  (τ is the time whose estimate is being updated)
        If τ ≥ 0:
            G ← Σ_{i=τ+1}^{min(τ+n,T)} γ^(i-τ-1) · Ri
            If τ + n < T:
                G ← G + γ^n · Q(Sτ+n, Aτ+n, w)
            w ← w + α[G - Q(Sτ, Aτ, w)]∇Q(Sτ, Aτ, w)
        If τ = T - 1: break
"""

class SemiGradientNStepSARSA:
    def __init__(self, feature_extractor: TileCoding, num_actions: int,
                 alpha: float = 0.01, gamma: float = 0.99,
                 n: int = 4, epsilon: float = 0.1):

        self.feature_extractor = feature_extractor
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.n = n
        self.epsilon = epsilon

        num_features = feature_extractor.get_num_features()
        self.w = np.zeros((num_actions, num_features))

    def get_q_value(self, state: np.ndarray, action: int) -> float:
        features = self.feature_extractor.get_features(state)
        return np.dot(self.w[action], features)

    def get_all_q_values(self, state: np.ndarray) -> np.ndarray:
        features = self.feature_extractor.get_features(state)
        return self.w @ features

    def select_action(self, state: np.ndarray) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            q_values = self.get_all_q_values(state)
            return np.argmax(q_values)

    def train_episode(self, env) -> float:
        states = []
        actions = []
        rewards = [0]

        state, _ = env.reset()
        action = self.select_action(state)
        states.append(state)
        actions.append(action)

        T = float('inf')
        t = 0
        total_reward = 0

        while True:
            if t < T:
                next_state, reward, terminated, truncated, _ = env.step(actions[t])
                done = terminated or truncated
                total_reward += reward

                states.append(next_state)
                rewards.append(reward)

                if done:
                    T = t + 1
                else:
                    next_action = self.select_action(next_state)
                    actions.append(next_action)

            tau = t - self.n + 1

            if tau >= 0:
                G = 0.0
                for i in range(tau + 1, min(tau + self.n, T) + 1):
                    G += (self.gamma ** (i - tau - 1)) * rewards[i]

                if tau + self.n < T:
                    q_bootstrap = self.get_q_value(states[tau + self.n],
                                                    actions[tau + self.n])
                    G += (self.gamma ** self.n) * q_bootstrap

                features = self.feature_extractor.get_features(states[tau])
                q_current = np.dot(self.w[actions[tau]], features)
                self.w[actions[tau]] += self.alpha * (G - q_current) * features

            if tau == T - 1:
                break

            t += 1

        return total_reward


### 3.3 REINFORCE with Baseline

#### Pseudocode
"""
Algorithm: REINFORCE with Baseline (episodic)
Input: policy π(a|s,θ), baseline b(s,w)
Parameters: step sizes αθ, αw

Initialize policy parameters θ and baseline weights w
For each episode:
    Generate episode S0, A0, R1, ..., ST-1, AT-1, RT following π(·|·,θ)
    For t = 0, 1, ..., T-1:
        Gt ← Σ_{k=t+1}^T γ^(k-t-1) · Rk
        δ ← Gt - b(St, w)
        w ← w + αw · δ · ∇b(St, w)
        θ ← θ + αθ · γ^t · δ · ∇ln π(At|St, θ)
"""

class REINFORCEWithBaseline:
    def __init__(self, feature_extractor: TileCoding, num_actions: int,
                 alpha_theta: float = 0.001, alpha_w: float = 0.01,
                 gamma: float = 0.99):

        self.feature_extractor = feature_extractor
        self.num_actions = num_actions
        self.alpha_theta = alpha_theta
        self.alpha_w = alpha_w
        self.gamma = gamma

        num_features = feature_extractor.get_num_features()
        self.w = np.zeros(num_features)
        self.theta = np.zeros((num_actions, num_features))

    def get_baseline(self, state: np.ndarray) -> float:
        features = self.feature_extractor.get_features(state)
        return np.dot(self.w, features)

    def get_action_preferences(self, state: np.ndarray) -> np.ndarray:
        features = self.feature_extractor.get_features(state)
        return self.theta @ features

    def get_policy(self, state: np.ndarray) -> np.ndarray:
        preferences = self.get_action_preferences(state)
        exp_prefs = np.exp(preferences - np.max(preferences))
        return exp_prefs / np.sum(exp_prefs)

    def select_action(self, state: np.ndarray) -> int:
        probs = self.get_policy(state)
        return np.random.choice(self.num_actions, p=probs)

    def train_episode(self, env) -> float:
        states = []
        actions = []
        rewards = []

        state, _ = env.reset()
        done = False

        while not done:
            states.append(state)
            action = self.select_action(state)
            actions.append(action)

            next_state, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)

            done = terminated or truncated
            state = next_state

        total_reward = sum(rewards)
        T = len(rewards)

        G = 0
        for t in range(T - 1, -1, -1):
            G = rewards[t] + self.gamma * G

            features = self.feature_extractor.get_features(states[t])
            baseline = np.dot(self.w, features)

            delta = G - baseline

            self.w += self.alpha_w * delta * features

            probs = self.get_policy(states[t])
            grad_ln_pi = np.outer(np.eye(self.num_actions)[actions[t]] - probs, features)
            self.theta += self.alpha_theta * (self.gamma ** t) * delta * grad_ln_pi

        return total_reward


## 4. Training and Evaluation Functions

def train_agent(agent, env, num_episodes: int, eval_interval: int = 10) -> Tuple[List[float], List[float]]:
    episode_rewards = []
    eval_rewards = []

    for episode in tqdm(range(num_episodes), desc="Training"):
        reward = agent.train_episode(env)
        episode_rewards.append(reward)

        if (episode + 1) % eval_interval == 0:
            avg_reward = np.mean(episode_rewards[-eval_interval:])
            eval_rewards.append(avg_reward)

    return episode_rewards, eval_rewards


def evaluate_agent(agent, env, num_episodes: int = 100) -> Tuple[float, float]:
    rewards = []

    for _ in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            if hasattr(agent, 'get_all_q_values'):
                q_values = agent.get_all_q_values(state)
                action = np.argmax(q_values)
            else:
                action = agent.select_action(state)

            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)

    return np.mean(rewards), np.std(rewards)


def smooth_curve(data: List[float], window_size: int = 50) -> np.ndarray:

    if len(data) < window_size:
        window_size = len(data)

    smoothed = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    return smoothed


## 5. Hyperparameter Tuning

### Approach
"""
For hyperparameter tuning, we use a combination of:
1. **Grid search** for discrete parameters (n-step, number of tilings)
2. **Manual tuning** based on initial runs for learning rates
3. **Standard values** from literature for discount factor (γ = 0.99)

Key hyperparameters to tune:
- **Learning rates** (α, αθ, αw): Affect convergence speed and stability
- **Feature representation**: Number of tilings and tiles affect generalization
- **Exploration** (ε for SARSA): Balance exploration vs exploitation
- **n-step** (for SARSA): Trade-off between bias and variance

### Tuning Strategy
1. Start with values from literature
2. Adjust learning rates by orders of magnitude
3. Fine-tune feature representation
4. Test multiple random seeds for robustness
"""


## 6. Experiments on MountainCar-v0


if __name__ == "__main__":
    env_mountaincar = gym.make('MountainCar-v0')

    mountaincar_features = TileCoding(
        num_tilings=8,
        tiles_per_dim=[8, 8],
        state_bounds=[[-1.2, 0.6], [-0.07, 0.07]]
    )

    print("MountainCar Environment:")
    print(f"State space: {env_mountaincar.observation_space}")
    print(f"Action space: {env_mountaincar.action_space}")
    print(f"Number of features: {mountaincar_features.get_num_features()}")

    ### 6.1 One-Step Actor-Critic on MountainCar

    ac_mountaincar = OneStepActorCritic(
        feature_extractor=mountaincar_features,
        num_actions=3,
        alpha_theta=0.001,
        alpha_w=0.1,
        gamma=1.0
    )

    print("\nTraining One-Step Actor-Critic on MountainCar...")
    ac_mc_rewards, ac_mc_eval = train_agent(ac_mountaincar, env_mountaincar,
                                             num_episodes=1000, eval_interval=10)

    ac_mc_mean, ac_mc_std = evaluate_agent(ac_mountaincar, env_mountaincar)
    print(f"Final Performance: {ac_mc_mean:.2f} ± {ac_mc_std:.2f}")

    ### 6.2 Semi-Gradient n-step SARSA on MountainCar

    sarsa_mountaincar = SemiGradientNStepSARSA(
        feature_extractor=mountaincar_features,
        num_actions=3,
        alpha=0.3 / 8,
        gamma=1.0,
        n=8,
        epsilon=0.0
    )

    print("\nTraining Semi-Gradient n-step SARSA on MountainCar...")
    sarsa_mc_rewards, sarsa_mc_eval = train_agent(sarsa_mountaincar, env_mountaincar,
                                                   num_episodes=1000, eval_interval=10)

    sarsa_mc_mean, sarsa_mc_std = evaluate_agent(sarsa_mountaincar, env_mountaincar)
    print(f"Final Performance: {sarsa_mc_mean:.2f} ± {sarsa_mc_std:.2f}")

    ### 6.3 REINFORCE with Baseline on MountainCar

    reinforce_mountaincar = REINFORCEWithBaseline(
        feature_extractor=mountaincar_features,
        num_actions=3,
        alpha_theta=0.001,
        alpha_w=0.1,
        gamma=1.0
    )

    print("\nTraining REINFORCE with Baseline on MountainCar...")
    reinforce_mc_rewards, reinforce_mc_eval = train_agent(reinforce_mountaincar,
                                                           env_mountaincar,
                                                           num_episodes=1000,
                                                           eval_interval=10)

    reinforce_mc_mean, reinforce_mc_std = evaluate_agent(reinforce_mountaincar, env_mountaincar)
    print(f"Final Performance: {reinforce_mc_mean:.2f} ± {reinforce_mc_std:.2f}")

    ## 7. Results and Analysis

    ### 7.1 Learning Curves

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    window = 50

    ac_smooth = smooth_curve(ac_mc_rewards, window)
    sarsa_smooth = smooth_curve(sarsa_mc_rewards, window)
    reinforce_smooth = smooth_curve(reinforce_mc_rewards, window)

    ax.plot(range(len(ac_smooth)), ac_smooth, label='One-Step Actor-Critic', linewidth=2.5, alpha=0.8)
    ax.plot(range(len(sarsa_smooth)), sarsa_smooth, label='Semi-Gradient n-step SARSA', linewidth=2.5, alpha=0.8)
    ax.plot(range(len(reinforce_smooth)), reinforce_smooth, label='REINFORCE with Baseline', linewidth=2.5, alpha=0.8)

    ax.set_xlabel('Episode', fontsize=14)
    ax.set_ylabel('Return (Smoothed)', fontsize=14)
    ax.set_title('MountainCar-v0 Learning Curves', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

    ### 7.2 Performance Comparison

    results_data = {
        'Algorithm': ['One-Step Actor-Critic', 'Semi-Gradient n-step SARSA', 'REINFORCE with Baseline'],
        'Mean Return': [ac_mc_mean, sarsa_mc_mean, reinforce_mc_mean],
        'Std Dev': [ac_mc_std, sarsa_mc_std, reinforce_mc_std]
    }

    results_df = pd.DataFrame(results_data)

    print("\nFINAL PERFORMANCE COMPARISON - MountainCar-v0")
    print(results_df.to_string(index=False))

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    algorithms = ['Actor-Critic', 'n-step SARSA', 'REINFORCE']
    x = np.arange(len(algorithms))
    width = 0.5

    mc_means = [ac_mc_mean, sarsa_mc_mean, reinforce_mc_mean]
    mc_stds = [ac_mc_std, sarsa_mc_std, reinforce_mc_std]

    bars = ax.bar(x, mc_means, width, yerr=mc_stds, capsize=7, alpha=0.85,
                color=['#1f77b4', '#ff7f0e', '#2ca02c'], edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Algorithm', fontsize=14)
    ax.set_ylabel('Average Return', fontsize=14)
    ax.set_title('MountainCar-v0: Final Performance Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=-200, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Baseline (-200)')
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    
    hyperparams_data = {
        'Algorithm': ['Actor-Critic', 'n-step SARSA', 'REINFORCE'],
        'α_θ / α': [0.001, 0.0375, 0.001],
        'α_w': [0.1, 'N/A', 0.1],
        'n-step': ['N/A', 8, 'N/A'],
        'ε': ['N/A', 0.0, 'N/A'],
        'γ': [1.0, 1.0, 1.0],
        'Episodes': [1000, 1000, 1000]
    }

    hyperparams_df = pd.DataFrame(hyperparams_data)

    print("\nHYPERPARAMETER SUMMARY - MountainCar-v0")
    print(hyperparams_df.to_string(index=False))

