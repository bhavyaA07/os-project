# main.py

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

from utils.process_generator import generate_processes
from training.environment import CPUSchedulingEnv
from algorithms.schedulers import fcfs, sjf, round_robin

# ------------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------------

NUM_PROCESSES    = 10
MAX_ARRIVAL_TIME = 20
MAX_BURST_TIME   = 15
MAX_PRIORITY     = 5
SEED             = 42
TRAIN_TIMESTEPS  = 30_000
QUANTUM          = 4


# ------------------------------------------------------------------
# 2. Generate Process Dataset
# ------------------------------------------------------------------

def generate_dataset() -> list:
    processes = generate_processes(
        num_processes    = NUM_PROCESSES,
        max_arrival_time = MAX_ARRIVAL_TIME,
        max_burst_time   = MAX_BURST_TIME,
        max_priority     = MAX_PRIORITY,
        seed             = SEED,
    )
    print("\n[INFO] Generated Processes:")
    print(f"{'ID':>4} {'Arrival':>8} {'Burst':>7} {'Priority':>9}")
    print("-" * 32)
    for p in processes:
        print(
            f"{p['id']:>4} {p['arrival_time']:>8} "
            f"{p['burst_time']:>7} {p['priority']:>9}"
        )
    return processes


# ------------------------------------------------------------------
# 3. Run Traditional Schedulers
# ------------------------------------------------------------------

def run_traditional_schedulers(processes: list) -> dict:
    print("\n[INFO] Running Traditional Schedulers...")

    results = {}

    results["FCFS"] = fcfs(processes)
    print(f"  [OK] FCFS        -> {results['FCFS']}")

    results["SJF"] = sjf(processes)
    print(f"  [OK] SJF         -> {results['SJF']}")

    results["Round Robin"] = round_robin(processes, quantum=QUANTUM)
    print(f"  [OK] Round Robin -> {results['Round Robin']}")

    return results


# ------------------------------------------------------------------
# 4. Train DQN Agent
# ------------------------------------------------------------------

def train_dqn_agent(processes: list):
    print("\n[INFO] Initialising RL Environment...")

    env = CPUSchedulingEnv(process_list=processes, max_processes=NUM_PROCESSES)

    print("  [INFO] Checking environment with SB3 env_checker...")
    check_env(env, warn=True)
    print("  [OK] Environment check passed.")

    print(f"\n[INFO] Training DQN agent for {TRAIN_TIMESTEPS:,} timesteps...")

    model = DQN(
        policy                 = "MlpPolicy",
        env                    = env,
        learning_rate          = 1e-3,
        buffer_size            = 10_000,
        learning_starts        = 500,
        batch_size             = 64,
        gamma                  = 0.99,
        target_update_interval = 500,
        train_freq             = 4,
        verbose                = 1,
        seed                   = SEED,
    )

    model.learn(total_timesteps=TRAIN_TIMESTEPS)
    print("  [OK] Training complete.")

    return model, env


# ------------------------------------------------------------------
# 5. Evaluate Trained DQN Agent
# ------------------------------------------------------------------

def evaluate_dqn_agent(model, processes: list) -> dict:
    print("\n[INFO] Evaluating trained DQN agent...")

    env = CPUSchedulingEnv(process_list=processes, max_processes=NUM_PROCESSES)
    obs, _ = env.reset()

    terminated   = False
    truncated    = False
    total_reward = 0.0
    steps        = 0
    max_steps    = NUM_PROCESSES * MAX_BURST_TIME * 3

    while not (terminated or truncated) and steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(int(action))
        total_reward += reward
        steps        += 1

    completed = env.completed_processes

    if not completed:
        print("  [WARN] No processes completed during evaluation.")
        return {
            "avg_waiting_time"    : 0.0,
            "avg_turnaround_time" : 0.0,
            "throughput"          : 0.0,
        }

    waiting_times    = [p.get("waiting_time",    0.0) for p in completed]
    turnaround_times = [p.get("turnaround_time", 0.0) for p in completed]
    total_time       = env.current_time if env.current_time > 0 else 1

    result = {
        "avg_waiting_time"    : float(np.mean(waiting_times)),
        "avg_turnaround_time" : float(np.mean(turnaround_times)),
        "throughput"          : len(completed) / total_time,
    }

    print(f"  [OK] DQN Result -> {result}")
    print(f"  Processes completed : {len(completed)} / {NUM_PROCESSES}")
    print(f"  Total reward        : {total_reward:.2f}")

    return result


# ------------------------------------------------------------------
# 6. Build Results DataFrame
# ------------------------------------------------------------------

def build_dataframe(traditional: dict, dqn_result: dict) -> pd.DataFrame:
    rows = []

    for algo_name, metrics in traditional.items():
        rows.append({
            "Algorithm"       : algo_name,
            "Waiting Time"    : round(metrics["avg_waiting_time"],    4),
            "Turnaround Time" : round(metrics["avg_turnaround_time"], 4),
            "Throughput"      : round(metrics["throughput"],          4),
        })

    rows.append({
        "Algorithm"       : "DQN (RL)",
        "Waiting Time"    : round(dqn_result["avg_waiting_time"],    4),
        "Turnaround Time" : round(dqn_result["avg_turnaround_time"], 4),
        "Throughput"      : round(dqn_result["throughput"],          4),
    })

    df = pd.DataFrame(rows)
    print("\n[INFO] Results Summary:")
    print(df.to_string(index=False))
    return df


# ------------------------------------------------------------------
# 7. Plot Results
# ------------------------------------------------------------------

def plot_results(df: pd.DataFrame) -> None:
    print("\n[INFO] Generating comparison plots...")

    algorithms = df["Algorithm"].tolist()
    x          = np.arange(len(algorithms))
    bar_width  = 0.5
    colors     = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "CPU Scheduling Algorithm Comparison",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )

    # (a) Waiting Time
    ax = axes[0]
    bars = ax.bar(x, df["Waiting Time"], width=bar_width, color=colors)
    ax.set_title("(a) Average Waiting Time", fontweight="bold")
    ax.set_ylabel("Time Units")
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, rotation=15, ha="right")
    ax.set_ylim(0, df["Waiting Time"].max() * 1.3 + 1)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    # (b) Turnaround Time
    ax = axes[1]
    bars = ax.bar(x, df["Turnaround Time"], width=bar_width, color=colors)
    ax.set_title("(b) Average Turnaround Time", fontweight="bold")
    ax.set_ylabel("Time Units")
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, rotation=15, ha="right")
    ax.set_ylim(0, df["Turnaround Time"].max() * 1.3 + 1)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    # (c) Throughput
    ax = axes[2]
    bars = ax.bar(x, df["Throughput"], width=bar_width, color=colors)
    ax.set_title("(c) Throughput (processes / unit time)", fontweight="bold")
    ax.set_ylabel("Throughput")
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, rotation=15, ha="right")
    ax.set_ylim(0, df["Throughput"].max() * 1.3 + 0.01)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig("scheduling_comparison.png", dpi=150, bbox_inches="tight")
    print("  [OK] Plot saved -> scheduling_comparison.png")
    plt.show()


# ------------------------------------------------------------------
# 8. Entry Point
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("   CPU Scheduling Simulation - RL vs Traditional")
    print("=" * 55)

    processes = generate_dataset()
    traditional_results = run_traditional_schedulers(processes)
    model, env = train_dqn_agent(processes)
    dqn_metrics = evaluate_dqn_agent(model, processes)
    df = build_dataframe(traditional_results, dqn_metrics)
    plot_results(df)

    print("\n[DONE] Simulation complete.")
    print("=" * 55)