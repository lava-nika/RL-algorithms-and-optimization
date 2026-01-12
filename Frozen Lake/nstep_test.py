import numpy as np
from nstep_sarsa_frozenlake import run_multiple_nstep_sarsa

def tune_nstep_sarsa():
    env_name = "FrozenLake-v1"
    map_name = "4x4"
    is_slippery = True
    n_runs = 20

    num_episodes = 10000
    max_steps = 1000
    base_seed = 42

    n_list = [1, 2, 4]
    alpha_list = [0.05, 0.1, 0.2, 0.3]
    eps_list = [0.05, 0.1, 0.2]
    gamma = 0.99

    results = []

    print("\n=========================================")
    print("   BEGIN HYPERPARAMETER SWEEP (SARSA)   ")
    print("=========================================")
    print(f"Environment       : {env_name} ({map_name})")
    print(f"Slippery          : {is_slippery}")
    print(f"n_runs            : {n_runs}")
    print(f"num_episodes      : {num_episodes}")
    print(f"max_steps_per_ep  : {max_steps}")
    print("=========================================\n")

    for n in n_list:
        for alpha in alpha_list:
            for eps in eps_list:

                print(f"\n---- Testing config ----")
                print(f"n = {n}")
                print(f"alpha = {alpha}")
                print(f"epsilon = {eps}")
                print(f"num_episodes = {num_episodes}")
                print(f"max_steps_per_episode = {max_steps}")
                print("------------------------")

                agents, mean_R, std_R, mean_S, std_S, all_R, all_S = run_multiple_nstep_sarsa(
                    n_runs=n_runs,
                    env_name=env_name,
                    map_name=map_name,
                    is_slippery=is_slippery,
                    n=n,
                    alpha=alpha,
                    gamma=gamma,
                    epsilon=eps,
                    num_episodes=num_episodes,
                    max_steps_per_episode=max_steps,
                    base_seed=base_seed,
                )

                avg_last500 = np.mean(mean_R[-500:])
                avg_all = np.mean(mean_R)

                print(f"Results: avg_return_all={avg_all:.3f}, avg_return_last500={avg_last500:.3f}")

                results.append({
                    "n": n,
                    "alpha": alpha,
                    "epsilon": eps,
                    "num_episodes": num_episodes,
                    "max_steps_per_episode": max_steps,
                    "avg_return_all": float(avg_all),
                    "avg_return_last500": float(avg_last500),
                })


    results.sort(key=lambda d: d["avg_return_last500"], reverse=True)

    print("\n\n========================")
    print("   TOP CONFIGURATIONS   ")
    print("========================")
    for r in results[:5]:
        print(r)


if __name__ == "__main__":
    tune_nstep_sarsa()
