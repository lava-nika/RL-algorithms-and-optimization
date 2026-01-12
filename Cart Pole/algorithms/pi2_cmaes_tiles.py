# PI2-CMA-ES algorithm with tile coding

import numpy as np
from .tile_coding import TileCoder


class PI2_CMAES_Tiles:
    def __init__(self, 
                 n_actions=2,
                 population_size=15,
                 lambda_=10.0,
                 elite_ratio=0.3,
                 sigma_init=0.5,
                 num_tilings=8,
                 tiles_per_dim=4):

        self.n_actions = n_actions
        self.population_size = population_size
        self.lambda_ = lambda_
        self.elite_ratio = elite_ratio
        self.sigma = sigma_init
        self.num_tilings = num_tilings
        self.tiles_per_dim = tiles_per_dim
        
        self.tile_coder = TileCoder(
            num_tilings=num_tilings,
            tiles_per_dim=tiles_per_dim
        )
        
        self.feature_dim = self.tile_coder.size
        self.param_shape = (self.feature_dim, n_actions)
        self.n_params = self.feature_dim * n_actions
        
        self.theta_mean = np.zeros(self.n_params)
        
        self.sigma_vec = np.ones(self.n_params) * sigma_init
        
        self.best_theta = self.theta_mean.copy()
        self.best_reward = -np.inf
        
        self.episode_rewards = []
        
    def get_features(self, state):
        features = np.zeros(self.feature_dim)
        active_tiles = self.tile_coder.get_tiles(state)
        for tile in active_tiles:
            features[tile] = 1.0
        return features
    
    def softmax_policy(self, theta, features):
        weights = theta.reshape(self.param_shape)
        logits = np.dot(features, weights)
        logits_stable = logits - np.max(logits)
        exp_logits = np.exp(logits_stable)
        probs = exp_logits / np.sum(exp_logits)
        return probs
    
    def select_action(self, theta, state):
        features = self.get_features(state)
        probs = self.softmax_policy(theta, features)
        action = np.random.choice(self.n_actions, p=probs)
        return action
    
    def sample_population(self):
        population = []
        for _ in range(self.population_size):
            noise = np.random.randn(self.n_params) * self.sigma_vec
            theta = self.theta_mean + noise
            population.append(theta)
        return population
    
    def evaluate_policy(self, theta, env, max_steps=500):
        state, _ = env.reset()
        total_reward = 0
        
        for _ in range(max_steps):
            action = self.select_action(theta, state)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        return total_reward
    
    def compute_pi2_weights(self, rewards):
        costs = -np.array(rewards)
        exp_weights = np.exp(-costs / self.lambda_)
        weights = exp_weights / np.sum(exp_weights)
        return weights
    
    def update_distribution(self, population, weights):
        population = np.array(population)
        
        new_mean = np.sum(population * weights[:, np.newaxis], axis=0)
    
        n_elite = max(1, int(self.elite_ratio * self.population_size))
        elite_indices = np.argsort(weights)[-n_elite:]
        elite_samples = population[elite_indices]
        
        centered_elite = elite_samples - self.theta_mean
        new_sigma_vec = np.std(centered_elite, axis=0)
        
        alpha = 0.2
        self.sigma_vec = (1 - alpha) * self.sigma_vec + alpha * (new_sigma_vec + 1e-8)
        
        self.theta_mean = new_mean
        
        self.sigma *= 0.995
    
    def train_generation(self, env):
        population = self.sample_population()
        
        rewards = []
        for theta in population:
            reward = self.evaluate_policy(theta, env)
            rewards.append(reward)
        
        max_reward = max(rewards)
        if max_reward > self.best_reward:
            self.best_reward = max_reward
            best_idx = np.argmax(rewards)
            self.best_theta = population[best_idx].copy()
        
        weights = self.compute_pi2_weights(rewards)
        
        self.update_distribution(population, weights)
        
        mean_reward = np.mean(rewards)
        return mean_reward, max_reward
    
    def train(self, env, n_generations=100, verbose=True):
        history = []
        
        if verbose:
            print(f"Training PI2-CMA-ES with tile coding for {n_generations} generations")
            print(f"Population size = {self.population_size}")
            print(f"Tile coding: {self.num_tilings} tilings * {self.tiles_per_dim} tiles/dim")
            print(f"Feature dimension: {self.feature_dim}")
        
        for gen in range(1, n_generations + 1):
            mean_reward, best_reward = self.train_generation(env)
            history.append((mean_reward, best_reward))
            
            self.episode_rewards.append(mean_reward)
            
            if verbose and gen % 10 == 0:
                avg_sigma = np.mean(self.sigma_vec)
                print(f"Generation {gen}/{n_generations}: "
                      f"Mean={mean_reward:.2f}, best={best_reward:.2f}, "
                      f"sigma_avg={avg_sigma:.4f}")
        
        if verbose:
            print(f"Training completed.")
            print(f"Best reward achieved = {self.best_reward:.2f}")
        
        return history
    
    def train_episode(self, env):
        mean_reward, _ = self.train_generation(env)
        return mean_reward
    
    def evaluate(self, env, n_episodes=100):
        rewards = []
        for _ in range(n_episodes):
            reward = self.evaluate_policy(self.best_theta, env)
            rewards.append(reward)
        
        return np.mean(rewards), np.std(rewards)
