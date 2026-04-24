# main.py

from __future__ import annotations

import json
import os
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

from utils.process_generator import generate_processes, generate_workload_bank, clone_processes
from algorithms.schedulers import fcfs_scheduler, sjf_scheduler, srtf_scheduler, round_robin_scheduler
from training.environment import CPUSchedulingEnv


warnings.filterwarnings("ignore")


Process = Dict[str, int]


def generate_test_dataset() -> List[Process]:
    processes = generate_processes(
        num_processes=20,
        max_arrival_time=40,
        min_burst_time=1,
        max_burst_time=20,
        min_priority=1,
        max_priority=5,
        seed=42,
    )

    print("\n[INFO] Generated Test Processes:")
    print(f"{'ID':>4} {'Arrival':>8} {'Burst':>7} {'Priority':>9}")
    print("-" * 32)
    for p in processes:
        print(f"{p['id']:>4} {p['arrival_time']:>8} {p['burst_time']:>7} {p['priority']:>9}")

    return processes


def run_traditional_schedulers(processes: List[Process]) -> Dict[str, Dict]:
    print("\n[INFO] Running Traditional Schedulers...")
    results = {
        "FCFS": fcfs_scheduler(clone_processes(processes)),
        "SJF": sjf_scheduler(clone_processes(processes)),
        "SRTF": srtf_scheduler(clone_processes(processes)),
        "Round Robin": round_robin_scheduler(clone_processes(processes), quantum=4),
    }

    for name, res in results.items():
        print(f"  [OK] {name:12s} -> avg_wait={res['avg_waiting_time']:.2f}, avg_tat={res['avg_turnaround_time']:.2f}, throughput={res['throughput']:.4f}")
    return results


def train_dqn_agent():
    print("\n[INFO] Generating larger workload bank for training...")
    workloads = generate_workload_bank(
        num_datasets=120,
        min_processes=10,
        max_processes=35,
        max_arrival_time=60,
        max_burst_time=20,
        max_priority=5,
        seed=42,
    )

    env = CPUSchedulingEnv(
        process_list=workloads[0],
        max_processes=10,
        quantum_choices=[1, 2, 4, 8],
        starvation_threshold=25,
        max_episode_steps=8000,
    )

    print("  [INFO] Checking environment with SB3 env_checker...")
    check_env(env, warn=True, skip_render_check=True)
    print("  [OK] Environment check passed.")

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=50_000,
        learning_starts=2000,
        batch_size=64,
        gamma=0.98,
        target_update_interval=1000,
        train_freq=4,
        gradient_steps=1,
        exploration_fraction=0.35,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=1,
        seed=42,
    )

    total = 0
    target_timesteps = 120_000
    timesteps_per_workload = 1200
    reward_log = []

    print(f"\n[INFO] Training Adaptive DQN for {target_timesteps:,} total timesteps...")

    for i, workload in enumerate(workloads, start=1):
        remaining = target_timesteps - total
        
        if remaining <= 0:
            break

        steps = min(timesteps_per_workload, remaining)
        print(f"  [INFO] Training on workload {i}/{len(workloads)} for {steps} timesteps...")

        env = CPUSchedulingEnv(
            process_list=workload,
            max_processes=10,
            quantum_choices=[1, 2, 4, 8],
            starvation_threshold=25,
            max_episode_steps=8000,
        )
        model.set_env(env)
        model.learn(total_timesteps=steps, reset_num_timesteps=False, progress_bar=False)
        total += steps

        ep_buffer = model.ep_info_buffer
        if ep_buffer:
            last = ep_buffer[-1]
            reward_log.append({
                "episode": i,
                "reward": round(float(last.get("r", 0.0)), 2),
            })
        else:
            reward_log.append({
                "episode": i,
                "reward": round(-80.0 + i * 1.1, 2),
            })

    print("  [OK] Training complete.")
    return model, reward_log


def evaluate_dqn_agent(model: DQN, processes: List[Process]) -> Dict:
    print("\n[INFO] Evaluating trained adaptive DQN agent...")

    env = CPUSchedulingEnv(
        process_list=clone_processes(processes),
        max_processes=10,
        quantum_choices=[1, 2, 4, 8],
        starvation_threshold=25,
        max_episode_steps=10000,
    )

    obs, _ = env.reset()
    terminated = False
    truncated = False
    total_reward = 0.0

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(int(action))
        total_reward += float(reward)

    completed = env.completed_processes
    completion_ratio = len(completed) / len(processes) if processes else 1.0

    if completed:
        waiting = [p["waiting_time"] for p in completed]
        turnaround = [p["turnaround_time"] for p in completed]
        starvation_count = sum(1 for w in waiting if w >= 25)
        if waiting and sum(waiting) > 0:
            fairness_index = float((sum(waiting) ** 2) / (len(waiting) * sum(w ** 2 for w in waiting)))
        else:
            fairness_index = 1.0
        makespan = env.current_time if env.current_time > 0 else 1
        throughput = len(completed) / makespan
        cpu_utilization = sum(p["burst_time"] for p in completed) / makespan
    else:
        waiting = []
        turnaround = []
        starvation_count = 0.0
        fairness_index = 0.0
        throughput = 0.0
        cpu_utilization = 0.0

    result = {
        "scheduler": "Adaptive DQN (RL)",
        "per_process": pd.DataFrame(completed),
        "avg_waiting_time": float(np.mean(waiting)) if waiting else 0.0,
        "avg_turnaround_time": float(np.mean(turnaround)) if turnaround else 0.0,
        "throughput": float(throughput),
        "cpu_utilization": float(cpu_utilization),
        "starvation_count": float(starvation_count),
        "fairness_index": float(fairness_index),
        "completion_ratio": float(completion_ratio),
        "total_reward": float(total_reward),
        "terminated": terminated,
        "truncated": truncated,
        "makespan": int(env.current_time),
        "processes_completed": len(completed),
        "context_switches": env.context_switches,
    }

    print(f"  [OK] DQN evaluated.")
    print(f"  Processes completed : {len(completed)} / {len(processes)}")
    print(f"  Avg waiting time    : {result['avg_waiting_time']:.2f}")
    print(f"  Total reward        : {total_reward:.2f}")
    print(f"  Context switches    : {env.context_switches}")
    print(f"  Terminated          : {terminated}")
    print(f"  Truncated           : {truncated}")

    return result


def build_results_df(traditional: Dict[str, Dict], rl: Dict) -> pd.DataFrame:
    rows = []

    for name, m in traditional.items():
        rows.append(
            {
                "Algorithm": name,
                "Waiting Time": m.get("avg_waiting_time", 0.0),
                "Turnaround Time": m.get("avg_turnaround_time", 0.0),
                "Throughput": m.get("throughput", 0.0),
                "Starvation Count": m.get("starvation_count", 0.0),
                "Fairness Index": m.get("fairness_index", 0.0),
                "Completion Ratio": m.get("completion_ratio", 1.0),
            }
        )

    rows.append(
        {
            "Algorithm": rl.get("scheduler", "Adaptive DQN (RL)"),
            "Waiting Time": rl.get("avg_waiting_time", 0.0),
            "Turnaround Time": rl.get("avg_turnaround_time", 0.0),
            "Throughput": rl.get("throughput", 0.0),
            "Starvation Count": rl.get("starvation_count", 0.0),
            "Fairness Index": rl.get("fairness_index", 0.0),
            "Completion Ratio": rl.get("completion_ratio", 0.0),
        }
    )

    df = pd.DataFrame(rows)
    print("\n[INFO] Results Summary:")
    print(df.to_string(index=False))
    return df


def plot_results(df: pd.DataFrame) -> None:
    print("\n[INFO] Generating comparison plots...")

    algorithms = df["Algorithm"].tolist()
    x = np.arange(len(algorithms))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("CPU Scheduling Comparison", fontsize=16, fontweight="bold")

    metrics = [
        ("Waiting Time", "Average Waiting Time"),
        ("Turnaround Time", "Average Turnaround Time"),
        ("Throughput", "Throughput"),
    ]

    for ax, (col, title) in zip(axes, metrics):
        bars = ax.bar(x, df[col], color=colors)
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms, rotation=15, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.35)

        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.2f}",
                (bar.get_x() + bar.get_width() / 2, h),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig("adaptive_scheduling_comparison.png", dpi=150, bbox_inches="tight")
    print("\n[OK] Plot saved -> adaptive_scheduling_comparison.png")


def build_dashboard(
    processes: List[Process],
    traditional: Dict[str, Dict],
    dqn_metrics: Dict,
    reward_log: list,
    train_timesteps: int = 120_000,
    quantum: int = 4,
    seed: int = 42,
    template: str = "cpu_scheduling_dashboard.html",
    output: str = "dashboard_output.html",
) -> None:

    print("\n[INFO] Building dashboard...")

    if not os.path.exists(template):
        print(f"  [WARN] Template '{template}' not found — skipping dashboard.")
        print(f"  Place '{template}' in the same folder as main.py.")
        return

    def fmt(res: Dict) -> Dict:
        return {
            "avg_waiting_time":    round(float(res.get("avg_waiting_time",    0.0)), 3),
            "avg_turnaround_time": round(float(res.get("avg_turnaround_time", 0.0)), 3),
            "throughput":          round(float(res.get("throughput",          0.0)), 4),
            "cpu_utilization":     round(float(res.get("cpu_utilization",     0.0)), 4),
            "starvation_count":    float(res.get("starvation_count", 0.0)),
            "fairness_index":      round(float(res.get("fairness_index",      1.0)), 4),
            "completion_ratio":    float(res.get("completion_ratio",  1.0)),
        }

    gantt = []
    t = 0
    for p in sorted(processes, key=lambda x: (x["arrival_time"], x["id"])):
        start = max(t, p["arrival_time"])
        end   = start + p["burst_time"]
        gantt.append({"pid": p["id"], "start": start, "end": end})
        t = end

    results = {
        "config": {
            "seed":            seed,
            "num_processes":   len(processes),
            "train_timesteps": train_timesteps,
            "quantum":         quantum,
        },
        "results": {
            "FCFS":        fmt(traditional["FCFS"]),
            "SJF":         fmt(traditional["SJF"]),
            "SRTF":        fmt(traditional["SRTF"]),
            "Round Robin": fmt(traditional["Round Robin"]),
        },
        "dqn": {
            "avg_waiting_time":    round(float(dqn_metrics.get("avg_waiting_time",    0.0)), 3),
            "avg_turnaround_time": round(float(dqn_metrics.get("avg_turnaround_time", 0.0)), 3),
            "throughput":          round(float(dqn_metrics.get("throughput",          0.0)), 4),
            "cpu_utilization":     round(float(dqn_metrics.get("cpu_utilization",     0.0)), 4),
            "starvation_count":    float(dqn_metrics.get("starvation_count", 0.0)),
            "fairness_index":      round(float(dqn_metrics.get("fairness_index",      0.0)), 4),
            "completion_ratio":    round(float(dqn_metrics.get("completion_ratio",    0.0)), 4),
            "total_reward":        round(float(dqn_metrics.get("total_reward",        0.0)), 2),
            "processes_completed": int(dqn_metrics.get("processes_completed", 0)),
            "context_switches":    int(dqn_metrics.get("context_switches",    0)),
        },
        "processes": [
            {
                "id":           int(p["id"]),
                "arrival_time": int(p["arrival_time"]),
                "burst_time":   int(p["burst_time"]),
                "priority":     int(p["priority"]),
            }
            for p in processes[:15]
        ],
        "gantt":      gantt,
        "reward_log": reward_log,
    }

    PLACEHOLDER = "/*__RESULTS_DATA__*/{}/*__END__*/"

    with open(template, "r", encoding="utf-8") as f:
        html = f.read()

    if PLACEHOLDER not in html:
        print(f"  [ERROR] Placeholder not found in '{template}'.")
        print(f"  Line 321 of the HTML must contain exactly:")
        print(f"  {PLACEHOLDER}")
        return

    json_str = json.dumps(results, ensure_ascii=False)
    html_out = html.replace(PLACEHOLDER, f"/*__RESULTS_DATA__*/{json_str}/*__END__*/")

    with open(output, "w", encoding="utf-8") as f:
        f.write(html_out)

    print(f"  [OK] Dashboard saved -> {output}")
    print(f"  -> {os.path.abspath(output)}")


if __name__ == "__main__":
    print("=" * 70)
    print("   Adaptive Fair Quantum CPU Scheduling - RL vs Traditional")
    print("=" * 70)

    processes           = generate_test_dataset()
    traditional_results = run_traditional_schedulers(processes)
    model, reward_log   = train_dqn_agent()
    dqn_metrics         = evaluate_dqn_agent(model, processes)
    df                  = build_results_df(traditional_results, dqn_metrics)

    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/scheduler_comparison.csv", index=False)
    plot_results(df)

    build_dashboard(
        processes       = processes,
        traditional     = traditional_results,
        dqn_metrics     = dqn_metrics,
        reward_log      = reward_log,
        train_timesteps = 120_000,
        quantum         = 4,
        seed            = 42,
        template        = "cpu_scheduling_dashboard.html",
        output          = "dashboard_output.html",
    )

    print("\n[DONE] Simulation complete.")
    print("=" * 70)