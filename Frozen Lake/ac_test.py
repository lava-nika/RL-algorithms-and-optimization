import numpy as np
from actor_critic_frozenlake import (
    run_multiple_actor_critic,
)


def tune_actor_critic():
    env_name = "FrozenLake-v1"
    map_name = "4x4"

    is_slippery = True

    n_runs = 20

    num_episodes = 10000
    max_steps = 1000
    base_seed = 42
    gamma = 0.99

    # Grids
    alpha_theta_list = [0.005, 0.01, 0.02, 0.05, 0.1]
    alpha_v_list = [0.1, 0.2, 0.3, 0.5]

    results = []

    print("\n===============================================")
    print("   BEGIN HYPERPARAMETER SWEEP (Actor-Critic)  ")
    print("===============================================")
    print(f"Environment          : {env_name} ({map_name})")
    print(f"Slippery             : {is_slippery}")
    print(f"n_runs               : {n_runs}")
    print(f"gamma                : {gamma}")
    print(f"num_episodes         : {num_episodes}")
    print(f"max_steps_per_ep     : {max_steps}")
    print("===============================================\n")

    for alpha_theta in alpha_theta_list:
        for alpha_v in alpha_v_list:

            print("\n---- Testing config ----")
            print(f"alpha_theta = {alpha_theta}")
            print(f"alpha_v     = {alpha_v}")
            print(f"num_episodes = {num_episodes}")
            print(f"max_steps_per_episode = {max_steps}")
            print("------------------------")

            (
                agents,
                mean_R,
                std_R,
                mean_S,
                std_S,
                all_R,
                all_S,
            ) = run_multiple_actor_critic(
                n_runs=n_runs,
                env_name=env_name,
                map_name=map_name,
                is_slippery=is_slippery,
                alpha_theta=alpha_theta,
                alpha_v=alpha_v,
                gamma=gamma,
                num_episodes=num_episodes,
                max_steps_per_episode=max_steps,
                base_seed=base_seed,
            )

            window = min(500, num_episodes)
            avg_last = float(np.mean(mean_R[-window:]))
            avg_all = float(np.mean(mean_R))

            print(
                f"Results: avg_return_all={avg_all:.3f}, "
                f"avg_return_last{window}={avg_last:.3f}"
            )

            results.append(
                {
                    "alpha_theta": alpha_theta,
                    "alpha_v": alpha_v,
                    "num_episodes": num_episodes,
                    "max_steps_per_episode": max_steps,
                    "avg_return_all": avg_all,
                    "avg_return_last_window": avg_last,
                    "window": window,
                }
            )

    results.sort(key=lambda d: d["avg_return_last_window"], reverse=True)

    print("\n\n========================")
    print("     TOP CONFIGURATIONS ")
    print("========================")
    for r in results[:5]:
        print(
            f"alpha_theta={r['alpha_theta']}, "
            f"alpha_v={r['alpha_v']}, "
            f"avg_all={r['avg_return_all']:.3f}, "
            f"avg_last{r['window']}={r['avg_return_last_window']:.3f}"
        )


if __name__ == "__main__":
    tune_actor_critic()
