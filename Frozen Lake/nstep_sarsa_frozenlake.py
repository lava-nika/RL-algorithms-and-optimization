import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt


def epsilon_greedy(q_values, epsilon):
    n_actions = q_values.shape[0]
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    max_q = np.max(q_values)
    best_actions = np.flatnonzero(q_values == max_q)
    return np.random.choice(best_actions)

# n-step SARSA Agent
class NStepSarsaAgent:
    def __init__(self, n_states, n_actions,
                 n=4, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.n = n
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Q-table: shape (n_states, n_actions)
        self.Q = np.zeros((n_states, n_actions), dtype=np.float64)

    def select_action(self, state):
        return epsilon_greedy(self.Q[state], self.epsilon)

    def q_value(self, state, action):
        return self.Q[state, action]

    def update_from_episode(self, states, actions, rewards):
        n = self.n
        gamma = self.gamma
        alpha = self.alpha
        T = len(rewards)

        # For tau = 0,1,...,T-1 compute G_tau^(n) and update Q
        for tau in range(T):
            # Computing n-step return G
            G = 0.0
            start = tau + 1
            end = min(tau + n, T)
            power = 0
            for k in range(start, end + 1):
                G += (gamma ** power) * rewards[k - 1]
                power += 1

            # If tau + n < T, bootstrap from Q(S_{tau+n}, A_{tau+n})
            if tau + n < T:
                s_boot = states[tau + n]
                a_boot = actions[tau + n]
                G += (gamma ** n) * self.q_value(s_boot, a_boot)

            s_tau = states[tau]
            a_tau = actions[tau]
            q_old = self.q_value(s_tau, a_tau)

            # Semi-gradient update:
            self.Q[s_tau, a_tau] += alpha * (G - q_old)

# Single-run training loop
def run_single_nstep_sarsa(
    env_name="FrozenLake-v1",
    map_name="4x4",
    is_slippery=True,
    n=4,
    alpha=0.1,
    gamma=0.99,
    epsilon=0.1,
    num_episodes=5000,
    max_steps_per_episode=1000,
    seed=None,
):
    env = gym.make(env_name, map_name=map_name, is_slippery=is_slippery)

    if seed is not None:
        np.random.seed(seed)
        try:
            env.reset(seed=seed)
        except TypeError:
            env.seed(seed)

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    agent = NStepSarsaAgent(
        n_states=n_states,
        n_actions=n_actions,
        n=n,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
    )

    returns_per_episode = np.zeros(num_episodes, dtype=np.float64)
    steps_per_episode = np.zeros(num_episodes, dtype=np.int32)

    for ep in range(num_episodes):
        out = env.reset()
        if isinstance(out, tuple):
            state, _info = out
        else:
            state = out

        states = [state]
        actions = []
        rewards = []

        done = False
        total_return = 0.0
        steps = 0

        while not done and steps < max_steps_per_episode:
            action = agent.select_action(state)
            step_out = env.step(action)

            if len(step_out) == 5:
                next_state, reward, terminated, truncated, _info = step_out
                done = terminated or truncated
            else:
                next_state, reward, done, _info = step_out

            states.append(next_state)
            actions.append(action)
            rewards.append(reward)

            total_return += reward
            steps += 1
            state = next_state

        # Update Q from this full episode
        agent.update_from_episode(states, actions, rewards)

        returns_per_episode[ep] = total_return
        steps_per_episode[ep] = steps

    env.close()
    return agent, returns_per_episode, steps_per_episode


# Multiple runs
def run_multiple_nstep_sarsa(
    n_runs=10,
    env_name="FrozenLake-v1",
    map_name="4x4",
    is_slippery=True,
    n=4,
    alpha=0.1,
    gamma=0.99,
    epsilon=0.1,
    num_episodes=5000,
    max_steps_per_episode=1000,
    base_seed=0,
):
    all_returns = np.zeros((n_runs, num_episodes), dtype=np.float64)
    all_steps = np.zeros((n_runs, num_episodes), dtype=np.float64)
    agents = []

    for run in range(n_runs):
        seed = None if base_seed is None else base_seed + run
        agent, returns, steps = run_single_nstep_sarsa(
            env_name=env_name,
            map_name=map_name,
            is_slippery=is_slippery,
            n=n,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            num_episodes=num_episodes,
            max_steps_per_episode=max_steps_per_episode,
            seed=seed,
        )
        agents.append(agent)
        all_returns[run] = returns
        all_steps[run] = steps

    mean_returns = all_returns.mean(axis=0)
    std_returns = all_returns.std(axis=0)
    mean_steps = all_steps.mean(axis=0)
    std_steps = all_steps.std(axis=0)

    return (
        agents,
        mean_returns,
        std_returns,
        mean_steps,
        std_steps,
        all_returns,
        all_steps,
    )

# Deriving value and greedy policy from table
def compute_state_values_and_policy(Q):
    V = np.max(Q, axis=1)
    pi = np.argmax(Q, axis=1)
    return V, pi


def print_frozenlake_value_and_policy(V, pi, map_name="4x4"):
    env = gym.make("FrozenLake-v1", map_name=map_name)
    desc = env.unwrapped.desc.astype(str)
    env.close()

    if map_name == "4x4":
        rows, cols = 4, 4
    elif map_name == "8x8":
        rows, cols = 8, 8
    else:
        raise ValueError("Unsupported FrozenLake map")

    def action_symbol(a):
        return {0: "←", 1: "↓", 2: "→", 3: "↑"}.get(a, "?")

    # Print Value Function
    print("Value Function:")
    for r in range(rows):
        row_vals = []
        for c in range(cols):
            s = r * cols + c
            row_vals.append(f"{V[s]:7.4f}")
        print("\t".join(row_vals))
    print()

    # Print Greedy Policy
    print("Greedy Policy:")
    for r in range(rows):
        row_acts = []
        for c in range(cols):
            s = r * cols + c
            a = pi[s]
            arrow = action_symbol(a)

            cell = desc[r, c]
            if cell == "H":
                row_acts.append(f"{arrow}(H)")
            elif cell == "G":
                row_acts.append("(G)")
            else:
                row_acts.append(arrow)

        print("\t".join(row_acts))
    print()

# Plotting
def moving_average(x, window_size):
    if window_size <= 1:
        return x

    window = np.ones(window_size) / window_size
    conv = np.convolve(x, window, mode="valid")
    pad_len = len(x) - len(conv)
    pad_values = np.full(pad_len, conv[0])
    return np.concatenate([pad_values, conv])



def plot_return_and_steps_smooth(
    mean_returns,
    mean_steps,
    title_suffix="",
    save_prefix=None,
    show=True,
    smooth_window=500,
):
    num_episodes = len(mean_returns)
    episodes = np.arange(1, num_episodes + 1)

    # Smoothing
    smooth_R = moving_average(mean_returns, smooth_window)
    smooth_S = moving_average(mean_steps, smooth_window)

    # Return vs Episode 
    plt.figure(figsize=(6, 4))
    plt.plot(episodes, smooth_R)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title(
        f"Episodic Semi-Gradient n-step SARSA Algorithm: Return vs Episode {title_suffix}\n"
    )
    plt.tight_layout()
    if save_prefix is not None:
        plt.savefig(f"{save_prefix}_return_smooth.png", dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

    # Steps vs Episode 
    plt.figure(figsize=(6, 4))
    plt.plot(episodes, smooth_S, color="red")
    plt.xlabel("Episode")
    plt.ylabel("Steps per Episode")
    plt.title(
        f"Episodic Semi-Gradient n-step SARSA Algorithm: Steps vs Episode {title_suffix}\n"
    )
    plt.tight_layout()
    if save_prefix is not None:
        plt.savefig(f"{save_prefix}_steps_smooth.png", dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

def summarize_returns_for_table(all_returns):
    per_run_means = all_returns.mean(axis=1)  

    mean_reward = per_run_means.mean()
    std_error = per_run_means.std(ddof=1) / np.sqrt(per_run_means.shape[0])
    best_reward = per_run_means.max()

    return mean_reward, std_error, best_reward



if __name__ == "__main__":

    # Hyperparameters
    HP = {
        "n_runs": 20,
        "env_name": "FrozenLake-v1",
        "map_name": "4x4",
        "is_slippery": True,
        "n": 2,
        "alpha": 0.05,
        "gamma": 0.99,
        "epsilon": 0.05,
        "num_episodes": 80000,
        "max_steps_per_episode": 1000,
        "base_seed": 42,
    }

    print("\n=======================")
    print(" Running n-step SARSA ")
    print("=======================\n")

    print("Hyperparameters used:")
    for k, v in HP.items():
        print(f"  {k}: {v}")
    print("\n")

    # Running multiple seeds
    agents, mean_R, std_R, mean_S, std_S, all_R, all_S = run_multiple_nstep_sarsa(
        n_runs=HP["n_runs"],
        env_name=HP["env_name"],
        map_name=HP["map_name"],
        is_slippery=HP["is_slippery"],
        n=HP["n"],
        alpha=HP["alpha"],
        gamma=HP["gamma"],
        epsilon=HP["epsilon"],
        num_episodes=HP["num_episodes"],
        max_steps_per_episode=HP["max_steps_per_episode"],
        base_seed=HP["base_seed"],
    )

    print("------------------------------------------------")
    print("Performance Summary")
    print("------------------------------------------------")

    # Average over all episodes 
    avg_return_all = np.mean(mean_R)
    avg_steps_all = np.mean(mean_S)

    print(f"Average return over all episodes:       {avg_return_all:.4f}")
    print(f"Average steps over all episodes:        {avg_steps_all:.4f}")
    print()

    # Final greedy policy and value function
    final_agent = agents[-1]
    V, pi = compute_state_values_and_policy(final_agent.Q)
    print_frozenlake_value_and_policy(V, pi, map_name=HP["map_name"])

    # Plotting
    plot_return_and_steps_smooth(
    mean_R,
    mean_S,
    title_suffix=f"(n={HP['n']}, slippery={HP['is_slippery']})",
    smooth_window=200,
    save_prefix=f"sarsa_frozenlake_n{HP['n']}_slip{HP['is_slippery']}",
    show=True
)
    
    # Summary
    mean_reward, std_error, best_reward = summarize_returns_for_table(all_R)
    print("Table summary for Semi-Gradient n-step SARSA:")
    print(f"  Mean reward (across runs): {mean_reward:.4f}")
    print(f"  Std. error (across runs):  {std_error:.4f}")
    print(f"  Best run mean reward:      {best_reward:.4f}")
    print()
