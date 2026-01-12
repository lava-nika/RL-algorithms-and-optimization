import numpy as np
import matplotlib.pyplot as plt

from nstep_sarsa_frozenlake import run_multiple_nstep_sarsa
from reinforce_baseline_frozenlake import run_multiple_reinforce_baseline
from actor_critic_frozenlake import run_multiple_actor_critic


def moving_average(x, window):
    if window <= 1:
        return x
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(x, kernel, mode="valid")


def main():
    n_runs = 20
    env_name = "FrozenLake-v1"
    map_name = "4x4"
    is_slippery = True
    gamma = 0.99
    num_episodes = 80000
    max_steps = 1000

    smooth_window = 200

    # 1) n-step SARSA
    print("Running n-step SARSA...")
    (_, mean_R_sarsa,
     _, mean_S_sarsa,
     _, _, _) = run_multiple_nstep_sarsa(
        n_runs=n_runs,
        env_name=env_name,
        map_name=map_name,
        is_slippery=is_slippery,
        n=2,
        alpha=0.05,
        gamma=gamma,
        epsilon=0.05,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps,
        base_seed=42,
    )

    # 2) REINFORCE with baseline
    print("Running REINFORCE with baseline...")
    (_, mean_R_reinf,
     _, mean_S_reinf,
     _, _, _) = run_multiple_reinforce_baseline(
        n_runs=n_runs,
        env_name=env_name,
        map_name=map_name,
        is_slippery=is_slippery,
        alpha_theta=0.1,
        alpha_v=0.2,
        gamma=gamma,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps,
        base_seed=42,
    )

    # 3) One-step Actor-Critic
    print("Running Actor-Critic...")
    (_, mean_R_ac,
     _, mean_S_ac,
     _, _, _) = run_multiple_actor_critic(
        n_runs=n_runs,
        env_name=env_name,
        map_name=map_name,
        is_slippery=is_slippery,
        alpha_theta=0.1,
        alpha_v=0.5,
        gamma=gamma,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps,
        base_seed=42,
    )

    # Smoothing
    episodes = np.arange(1, num_episodes + 1)

    smooth_sarsa_R  = moving_average(mean_R_sarsa,  smooth_window)
    smooth_reinf_R  = moving_average(mean_R_reinf,  smooth_window)
    smooth_ac_R     = moving_average(mean_R_ac,     smooth_window)

    smooth_sarsa_S  = moving_average(mean_S_sarsa,  smooth_window)
    smooth_reinf_S  = moving_average(mean_S_reinf,  smooth_window)
    smooth_ac_S     = moving_average(mean_S_ac,     smooth_window)

    episodes_smooth = episodes[smooth_window - 1 :]

    # Plot 1: Return vs Episode
    plt.figure(figsize=(8, 5))
    plt.plot(episodes_smooth, smooth_sarsa_R, label="n-step SARSA (n=2)", linewidth=1.5)
    plt.plot(episodes_smooth, smooth_reinf_R, label="REINFORCE + baseline", linewidth=1.5)
    plt.plot(episodes_smooth, smooth_ac_R,    label="Actor-Critic", linewidth=1.5)

    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Learning Curves (Return vs Episode)")
    plt.legend()
    plt.tight_layout()

    plt.savefig("combined_returns.png", dpi=300)
    print("Saved: combined_returns.png")

    plt.show()

    # Plot 2: Steps vs Episode
    plt.figure(figsize=(8, 5))
    plt.plot(episodes_smooth, smooth_sarsa_S, label="Semi-Gradient n-step SARSA", linewidth=1.5)
    plt.plot(episodes_smooth, smooth_reinf_S, label="REINFORCE + baseline", linewidth=1.5)
    plt.plot(episodes_smooth, smooth_ac_S,    label="One-Step Actor-Critic", linewidth=1.5)

    plt.xlabel("Episode")
    plt.ylabel("Steps per Episode")
    plt.title("Learning Curves (Steps vs Episode)")
    plt.legend()
    plt.tight_layout()

    plt.savefig("combined_steps.png", dpi=300)
    print("Saved: combined_steps.png")

    plt.show()


if __name__ == "__main__":
    main()
