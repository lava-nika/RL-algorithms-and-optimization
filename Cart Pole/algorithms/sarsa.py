# Semi-gradient n-step SARSA algorithm with tile coding
import numpy as np
from collections import deque
from .tile_coding import TileCoder

class SemiGradientNStepSarsa:
    def __init__(self, n_actions=2, n_steps=4, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.n_actions = n_actions
        self.n_steps = n_steps
        self.alpha = alpha / 8
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.tile_coder = TileCoder(num_tilings=8, tiles_per_dim=8)
        
        self.w = np.zeros(self.tile_coder.size * n_actions)
    
    def get_q_value(self, state, action):
        tiles = self.tile_coder.get_tiles(state)
        offset = action * self.tile_coder.size
        return sum(self.w[t + offset] for t in tiles)
    
    def select_action(self, state, explore=True):
        if explore and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        
        q_values = [self.get_q_value(state, a) for a in range(self.n_actions)]
        return np.argmax(q_values)
    
    def train_episode(self, env):
        states = deque(maxlen=self.n_steps + 1)
        actions = deque(maxlen=self.n_steps + 1)
        rewards = deque(maxlen=self.n_steps)
        
        state, _ = env.reset()
        action = self.select_action(state)
        
        states.append(state)
        actions.append(action)
        
        T = float('inf')
        t = 0
        total_reward = 0
        
        while True:
            if t < T:
                next_state, reward, done, truncated, _ = env.step(action)
                done = done or truncated
                total_reward += reward
                rewards.append(reward)
                
                if done:
                    T = t + 1
                else:
                    next_action = self.select_action(next_state)
                    states.append(next_state)
                    actions.append(next_action)
                    action = next_action
            
            tau = t - self.n_steps + 1
            
            if tau >= 0:
                G = sum([self.gamma ** (i - tau - 1) * rewards[i - tau - 1] 
                        for i in range(tau + 1, min(tau + self.n_steps, T) + 1)])
                
                if tau + self.n_steps < T:
                    G += self.gamma ** self.n_steps * self.get_q_value(
                        states[self.n_steps], actions[self.n_steps])
                
                tiles = self.tile_coder.get_tiles(states[0])
                offset = actions[0] * self.tile_coder.size
                q_current = sum(self.w[t + offset] for t in tiles)
                
                for tile in tiles:
                    self.w[tile + offset] += self.alpha * (G - q_current)
            
            if tau == T - 1:
                break
            
            t += 1
        
        return total_reward