# REINFORCE with baseline algorithm
import numpy as np

class REINFORCE:
    def __init__(self, state_dim=4, n_actions=2, alpha_theta=0.001, 
                 alpha_w=0.01, gamma=0.99):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.alpha_theta = alpha_theta
        self.alpha_w = alpha_w
        self.gamma = gamma
        
        self.theta = np.random.randn(state_dim + 1, n_actions) * 0.01
        
        self.w = np.random.randn(state_dim + 1) * 0.01
    
    def add_bias(self, state):
        return np.append(state, 1.0)
    
    def get_action_probs(self, state):
        state = self.add_bias(state)
        h = state @ self.theta
        exp_h = np.exp(h - np.max(h))
        return exp_h / np.sum(exp_h)
    
    def select_action(self, state):
        probs = self.get_action_probs(state)
        return np.random.choice(self.n_actions, p=probs)
    
    def get_value(self, state):
        state = self.add_bias(state)
        return state @ self.w
    
    def train_episode(self, env):
        states = []
        actions = []
        rewards = []
        
        state, _ = env.reset()
        done = False
        
        while not done:
            action = self.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated 
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            state = next_state
        
        total_reward = sum(rewards)
        
        G = 0
        for t in reversed(range(len(states))):
            G = rewards[t] + self.gamma * G
            state_t = self.add_bias(states[t])
            action_t = actions[t]
            
            v = state_t @ self.w
            delta = G - v
            self.w += self.alpha_w * delta * state_t
            
            probs = self.get_action_probs(states[t])
            
            for a in range(self.n_actions):
                if a == action_t:
                    grad = state_t * (1.0 - probs[a])
                else:
                    grad = -state_t * probs[a]
                
                self.theta[:, a] += self.alpha_theta * delta * grad
        
        return total_reward