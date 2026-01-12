import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

def softmax(logits):
    z = logits - np.max(logits)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)


# REINFORCE with baseline Agent
class ReinforceBaselineAgent:
    def __init__(self, n_states, n_actions,
                 alpha_theta=0.05, alpha_v=0.1, gamma=0.99):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha_theta = alpha_theta
        self.alpha_v = alpha_v
        self.gamma = gamma

        # Policy parameters: one logit per (s,a)
        self.theta = np.zeros((n_states, n_actions), dtype=np.float64)

        # Baseline parameters: one value per state
        self.V = np.zeros(n_states, dtype=np.float64)

    def policy(self, state):
        logits = self.theta[state]
        return softmax(logits)

    def select_action(self, state):
        probs = self.policy(state)
        return np.random.choice(self.n_actions, p=probs)

    def update_from_episode(self, states, actions, rewards):
        gamma = self.gamma
        alpha_theta = self.alpha_theta
        alpha_v = self.alpha_v
        T = len(rewards)

        # Precomputing reward-to-go returns G_t for t = 0,...,T-1
        G = np.zeros(T, dtype=np.float64)
        G_t = 0.0
        for t in reversed(range(T)):
            G_t = rewards[t] + gamma * G_t
            G[t] = G_t

        # Update for each time step t
        for t in range(T):
            s_t = states[t]
            a_t = actions[t]
            G_t = G[t]

            # Baseline value
            v_s = self.V[s_t]
            advantage = G_t - v_s

            # Critic update (baseline)
            self.V[s_t] += alpha_v * (G_t - v_s)

            # Actor update
            probs = self.policy(s_t)
            grad_log_pi = -probs
            grad_log_pi[a_t] += 1.0

            self.theta[s_t] += alpha_theta * advantage * grad_log_pi


# Single-run training loop
def run_single_reinforce_baseline(
    env_name="FrozenLake-v1",
    map_name="4x4",
    is_slippery=True,
    alpha_theta=0.05,
    alpha_v=0.1,
    gamma=0.99,
    num_episodes=5000,
    max_steps_per_episode=1000,
    seed=None,
):
    env = gym.make(env_name, map_name=map_name, is_slippery=is_slippery)

    # Seeding
    if seed is not None:
        np.random.seed(seed)
        try:
            env.reset(seed=seed)
        except TypeError:
            env.seed(seed)

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    agent = ReinforceBaselineAgent(
        n_states=n_states,
        n_actions=n_actions,
        alpha_theta=alpha_theta,
        alpha_v=alpha_v,
        gamma=gamma,
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

        # Monte Carlo policy gradient update
        agent.update_from_episode(states, actions, rewards)

        returns_per_episode[ep] = total_return
        steps_per_episode[ep] = steps

    env.close()
    return agent, returns_per_episode, steps_per_episode


# Multiple runs 
def run_multiple_reinforce_baseline(
    n_runs=10,
    env_name="FrozenLake-v1",
    map_name="4x4",
    is_slippery=True,
    alpha_theta=0.05,
    alpha_v=0.1,
    gamma=0.99,
    num_episodes=5000,
    max_steps_per_episode=1000,
    base_seed=0,
):
    all_returns = np.zeros((n_runs, num_episodes), dtype=np.float64)
    all_steps = np.zeros((n_runs, num_episodes), dtype=np.float64)
    agents = []

    for run in range(n_runs):
        seed = None if base_seed is None else base_seed + run
        agent, returns, steps = run_single_reinforce_baseline(
            env_name=env_name,
            map_name=map_name,
            is_slippery=is_slippery,
            alpha_theta=alpha_theta,
            alpha_v=alpha_v,
            gamma=gamma,
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

def compute_state_values_and_greedy_policy_theta(agent):
    n_states = agent.n_states
    n_actions = agent.n_actions

    V = agent.V.copy()
    pi = np.zeros(n_states, dtype=np.int32)

    for s in range(n_states):
        probs = softmax(agent.theta[s])
        pi[s] = np.argmax(probs)

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

    print("Value Function:")
    for r in range(rows):
        row_vals = []
        for c in range(cols):
            s = r * cols + c
            row_vals.append(f"{V[s]:7.4f}")
        print("\t".join(row_vals))
    print()

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


def moving_average(x, window_size):
    if window_size <= 1:
        return x

    window = np.ones(window_size) / window_size
    conv = np.convolve(x, window, mode="valid")
    pad_len = len(x) - len(conv)
    pad_values = np.full(pad_len, conv[0])
    return np.concatenate([pad_values, conv])


# Plotting
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
        f"REINFORCE with Baseline Algorithm: Return vs Episode {title_suffix}\n"
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
        f"REINFORCE with Baseline Algorithm: Steps vs Episode {title_suffix}\n"
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

    HP = {
        "n_runs": 20,
        "env_name": "FrozenLake-v1",
        "map_name": "4x4",
        "is_slippery": True,
        "alpha_theta": 0.1,
        "alpha_v": 0.2,
        "gamma": 0.99,
        "num_episodes": 80000,
        "max_steps_per_episode": 1000,
        "base_seed": 42,
    }

    print("\n=======================================")
    print(" Running REINFORCE with baseline ")
    print("=======================================\n")

    print("Hyperparameters used:")
    for k, v in HP.items():
        print(f"  {k}: {v}")
    print()

    agents, mean_R, std_R, mean_S, std_S, all_R, all_S = run_multiple_reinforce_baseline(
        n_runs=HP["n_runs"],
        env_name=HP["env_name"],
        map_name=HP["map_name"],
        is_slippery=HP["is_slippery"],
        alpha_theta=HP["alpha_theta"],
        alpha_v=HP["alpha_v"],
        gamma=HP["gamma"],
        num_episodes=HP["num_episodes"],
        max_steps_per_episode=HP["max_steps_per_episode"],
        base_seed=HP["base_seed"],
    )

    print("------------------------------------------------")
    print("Performance Summary")
    print("------------------------------------------------")

    avg_return_all = np.mean(mean_R)
    avg_steps_all = np.mean(mean_S)

    print(f"Average return over ALL episodes:       {avg_return_all:.4f}")
    print(f"Average steps over ALL episodes:        {avg_steps_all:.4f}")
    print()

    final_agent = agents[-1]
    V, pi = compute_state_values_and_greedy_policy_theta(final_agent)
    print_frozenlake_value_and_policy(V, pi, map_name=HP["map_name"])

    plot_return_and_steps_smooth(
        mean_R,
        mean_S,
        title_suffix=f"(slippery={HP['is_slippery']})",
        save_prefix="reinforce_baseline_frozenlake",
        show=True,
        smooth_window=200,
    )

    # Summary
    mean_reward, std_error, best_reward = summarize_returns_for_table(all_R)
    print("Table summary for REINFORCE with Baseline:")
    print(f"  Mean reward (across runs): {mean_reward:.4f}")
    print(f"  Std. error (across runs):  {std_error:.4f}")
    print(f"  Best run mean reward:      {best_reward:.4f}")
    print()
