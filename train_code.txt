import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from autoscaling_env import MultiTierAutoscalingEnv, load_workload

def discretize(obs):
    # compact tabular state for Q-Learning
    bins = [np.array([0.15, 0.35, 0.60, 0.85]),
            np.array([0.45, 0.70, 0.90, 1.10, 1.40]),
            np.array([0.45, 0.70, 0.90, 1.10, 1.40]),
            np.array([0.45, 0.70, 0.90, 1.10, 1.40]),
            np.array([0.30, 0.60, 1.00, 1.50])]
    return tuple(int(np.digitize(float(x), b)) for x, b in zip(obs, bins))

def train_q_learning(workload, episodes=25, seed=42):
    rng = np.random.default_rng(seed)
    env = MultiTierAutoscalingEnv(workload, max_steps=min(1440, len(workload)), seed=seed)
    q = {}
    alpha, gamma = 0.12, 0.95
    for ep in range(episodes):
        obs, _ = env.reset()
        state = discretize(obs)
        done = False
        epsilon = max(0.05, 0.9 * (0.94 ** ep))
        while not done:
            if state not in q:
                q[state] = np.zeros(env.action_space.n)
            action = int(rng.integers(env.action_space.n)) if rng.random() < epsilon else int(np.argmax(q[state]))
            nxt, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            ns = discretize(nxt)
            if ns not in q:
                q[ns] = np.zeros(env.action_space.n)
            q[state][action] += alpha * (reward + gamma * (0 if done else np.max(q[ns])) - q[state][action])
            state = ns
    return q

def evaluate_q(workload, q, seed=100):
    env = MultiTierAutoscalingEnv(workload, max_steps=min(1440, len(workload)), seed=seed)
    obs, _ = env.reset()
    rows, done = [], False
    while not done:
        state = discretize(obs)
        action = int(np.argmax(q.get(state, np.zeros(env.action_space.n))))
        obs, _, term, trunc, info = env.step(action)
        rows.append(info)
        done = term or trunc
    return rows

def evaluate_sb3(model, workload, seed=100):
    env = MultiTierAutoscalingEnv(workload, max_steps=min(1440, len(workload)), seed=seed)
    obs, _ = env.reset()
    rows, done = [], False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(int(action))
        rows.append(info)
        done = term or trunc
    return rows

def evaluate_hpa(workload, seed=100):
    env = MultiTierAutoscalingEnv(workload, max_steps=min(1440, len(workload)), seed=seed)
    obs, _ = env.reset()
    rows, done = [], False
    while not done:
        # Threshold controller: scale the tier with the highest utilization.
        util = obs[1:4]
        tier = int(np.argmax(util))
        if util[tier] > 0.75:
            action = 1 + tier * 2
        elif util[tier] < 0.35:
            action = 2 + tier * 2
        else:
            action = 0
        obs, _, term, trunc, info = env.step(action)
        rows.append(info)
        done = term or trunc
    return rows

def summarize(name, rows):
    df = pd.DataFrame(rows)
    return {
        "algorithm": name,
        "mean_response_time_ms": df.response_time_ms.mean(),
        "sla_violation_rate_pct": 100 * df.sla_violation.mean(),
        "mean_resource_cost": df.cost.mean(),
        "mean_total_replicas": df.total_replicas.mean(),
        "mean_cpu_utilization_pct": df.avg_cpu_pct.mean(),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--ppo-steps", type=int, default=12000)
    parser.add_argument("--trpo-steps", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    Path("results").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    workload = load_workload()
    # Keep training/evaluation practical for a laptop; trace remains NASA-derived.
    workload = workload[:min(len(workload), 5000)]
    print(f"NASA-derived workload intervals: {len(workload)}; p95={np.percentile(workload,95):.3f} req/s")

    q = train_q_learning(workload, args.episodes, args.seed)
    np.save("models/q_table.npy", q, allow_pickle=True)
    q_rows = evaluate_q(workload, q)

    from stable_baselines3 import PPO
    from sb3_contrib import TRPO

    def make_env():
        return MultiTierAutoscalingEnv(workload, max_steps=min(1440, len(workload)), seed=args.seed)

    ppo_env = make_env()
    ppo = PPO("MlpPolicy", ppo_env, verbose=0, seed=args.seed, n_steps=256, batch_size=64)
    ppo.learn(total_timesteps=args.ppo_steps)
    ppo.save("models/ppo_autoscaler")
    ppo_rows = evaluate_sb3(ppo, workload)

    trpo_env = make_env()
    trpo = TRPO("MlpPolicy", trpo_env, verbose=0, seed=args.seed, n_steps=256, batch_size=64)
    trpo.learn(total_timesteps=args.trpo_steps)
    trpo.save("models/trpo_autoscaler")
    trpo_rows = evaluate_sb3(trpo, workload)

    hpa_rows = evaluate_hpa(workload)
    all_rows = []
    for name, rows in [("Q-Learning", q_rows), ("PPO", ppo_rows), ("TRPO", trpo_rows), ("HPA baseline", hpa_rows)]:
        df = pd.DataFrame(rows)
        df["algorithm"] = name
        df["step"] = np.arange(len(df))
        df.to_csv(f"results/{name.lower().replace(' ', '_').replace('-', '')}_timeseries.csv", index=False)
        all_rows.append(summarize(name, rows))
    summary = pd.DataFrame(all_rows)
    summary.to_csv("results/summary.csv", index=False)
    print("\nRESULTS SUMMARY\n", summary.to_string(index=False))

    # Graph 1: response time
    plt.figure(figsize=(10, 5))
    for name, rows in [("Q-Learning", q_rows), ("PPO", ppo_rows), ("TRPO", trpo_rows), ("HPA baseline", hpa_rows)]:
        plt.plot(pd.DataFrame(rows).response_time_ms.rolling(10, min_periods=1).mean(), label=name)
    plt.axhline(500, linestyle="--", label="SLA threshold (500 ms)")
    plt.xlabel("Evaluation interval (minutes)")
    plt.ylabel("Response time (ms)")
    plt.title("Response Time Comparison (10-interval moving average)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/response_time_comparison.png", dpi=160)
    plt.close()

    # Graph 2: resource cost and replicas
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(summary.algorithm, summary.mean_resource_cost)
    ax.set_ylabel("Mean estimated resource cost / interval")
    ax.set_title("Estimated Resource Cost Comparison")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig("results/resource_cost_comparison.png", dpi=160)
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(summary.algorithm, summary.sla_violation_rate_pct)
    ax.set_ylabel("SLA violation rate (%)")
    ax.set_title("SLA Violation Comparison")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig("results/sla_violation_comparison.png", dpi=160)
    plt.close()
    print("\nSaved summary, time-series CSVs, and three graphs in results/.")

if __name__ == "__main__":
    main()
