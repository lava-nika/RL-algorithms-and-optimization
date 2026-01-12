# CartPole-v1 Environment

Comparative analysis of 4 RL algorithms on CartPole-v1: Semi-gradient n-step SARSA, REINFORCE with baseline, One-step Actor-Critic, and PI²-CMA-ES.

**Best Result:** PI²-CMA-ES achieved **499.87 ± 0.57**.

---
## CartPole-v1 overview

**Problem Description:**  
CartPole-v1 is a classic control benchmark from OpenAI Gymnasium that simulates balancing a pole attached to a cart moving along a frictionless track. The agent must learn to balance the pole upright by applying horizontal forces to the cart.

**State Space (4 dimensions):**
- **Cart Position** (x): Position of cart on track, range approximately [-2.4, 2.4]
- **Cart Velocity** (ẋ): Velocity of cart, range approximately [-∞, ∞]
- **Pole Angle** (θ): Angle of pole from vertical (0° = upright), range approximately [-0.209, 0.209] radians (~±12°)
- **Pole Angular Velocity** (θ̇): Rate of change of pole angle, range approximately [-∞, ∞]

**Action Space (2 discrete actions):**
- **Action 0**: Push cart to the **left** (negative force)
- **Action 1**: Push cart to the **right** (positive force)

**Rewards:**
- **+1 reward** for every timestep the pole remains upright
- Goal: Maximize cumulative reward by keeping pole balanced as long as possible

**Episode Termination:**
An episode ends when any of the following occurs:
1. **Pole angle** exceeds ±12° from vertical (|θ| > 0.209 rad)
2. **Cart position** exceeds ±2.4 units from center (|x| > 2.4)
3. **Maximum episode length** of 500 timesteps is reached

**Success Criteria:**  
The environment is considered **solved** when the agent achieves an average reward of **≥475.0** over 100 consecutive episodes, meaning it can balance the pole for nearly the full 500 timesteps consistently.

**Physical Intuition:**  
The agent must learn to:
1. Move cart in the direction the pole is falling to "catch" it
2. Anticipate future pole motion using velocity information
3. Balance between pole angle correction and cart position constraints
4. Avoid oscillating too aggressively (which destabilizes the system)

---

## Requirements

- Python 3.8+
- numpy
- matplotlib
- gymnasium

--- 

## Instructions to run 

### 1. Setup Environment

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install numpy matplotlib gymnasium
```

### 2. Run Training

```bash
# Train all algorithms with optimal hyperparameters
python3 main.py
```

This will train all the 4 algorithms, generate learning curves and print final performance statistics.

---

## Main project files

```
cartpole/
├── main.py                          # Main training script
├── algorithms/
│   ├── sarsa.py                     # Semi-gradient n-step SARSA
│   ├── reinforce.py                 # REINFORCE with baseline
│   ├── actor_critic.py              # One-step Actor-Critic
│   └── pi2_cmaes_tiles.py          # PI²-CMA-ES with tile coding
└── tuning_experiments/              # Hyperparameter tuning scripts
```
---

