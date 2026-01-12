# One-step Actor-Critic algorithm 
import numpy as np

class ActorCritic:
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
        state, _ = env.reset()
        done = False
        total_reward = 0
        I = 1
        
        while not done:
            action = self.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            v_current = self.get_value(state)
            v_next = 0 if done else self.get_value(next_state)
            delta = reward + self.gamma * v_next - v_current
            
            state_bias = self.add_bias(state)
            self.w += self.alpha_w * delta * state_bias
            
            probs = self.get_action_probs(state)
            
            for a in range(self.n_actions):
                if a == action:
                    grad = state_bias * (1.0 - probs[a])
                else:
                    grad = -state_bias * probs[a]
                
                self.theta[:, a] += self.alpha_theta * I * delta * grad
            
            I *= self.gamma
            state = next_state
        
        return total_reward