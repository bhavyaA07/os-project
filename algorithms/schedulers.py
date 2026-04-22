# algorithms/schedulers.py

from __future__ import annotations

from typing import Dict, List
import pandas as pd


Process = Dict[str, int]


def _clone_processes(processes: List[Process]) -> List[Process]:
    return [
        {
            "id": int(p["id"]),
            "arrival_time": int(p["arrival_time"]),
            "burst_time": int(p["burst_time"]),
            "remaining_time": int(p["remaining_time"]),
            "priority": int(p["priority"]),
        }
        for p in processes
    ]


def _build_result(name: str, original: List[Process], completion_times: Dict[int, int]) -> Dict:
    rows = []
    for p in sorted(original, key=lambda x: x["id"]):
        ct = completion_times[p["id"]]
        tat = ct - p["arrival_time"]
        wt = tat - p["burst_time"]
        rows.append(
            {
                "id": p["id"],
                "arrival_time": p["arrival_time"],
                "burst_time": p["burst_time"],
                "priority": p["priority"],
                "completion_time": ct,
                "turnaround_time": tat,
                "waiting_time": wt,
            }
        )

    df = pd.DataFrame(rows)
    makespan = int(df["completion_time"].max()) if not df.empty else 1
    total_burst = int(df["burst_time"].sum()) if not df.empty else 0

    avg_wait = float(df["waiting_time"].mean()) if not df.empty else 0.0
    avg_tat = float(df["turnaround_time"].mean()) if not df.empty else 0.0
    throughput = float(len(df) / makespan) if makespan > 0 else 0.0
    cpu_utilization = float(total_burst / makespan) if makespan > 0 else 0.0

    return {
        "scheduler": name,
        "per_process": df,
        "avg_waiting_time": avg_wait,
        "avg_turnaround_time": avg_tat,
        "throughput": throughput,
        "cpu_utilization": cpu_utilization,
        "starvation_count": 0.0,
        "fairness_index": 1.0 / (1.0 + (df["waiting_time"].max() - df["waiting_time"].min())) if not df.empty else 1.0,
        "completion_ratio": 1.0,
        "makespan": makespan,
    }


def fcfs_scheduler(processes: List[Process]) -> Dict:
    procs = _clone_processes(processes)
    original = _clone_processes(processes)

    time = 0
    completion_times: Dict[int, int] = {}

    for p in sorted(procs, key=lambda x: (x["arrival_time"], x["id"])):
        if time < p["arrival_time"]:
            time = p["arrival_time"]
        time += p["burst_time"]
        completion_times[p["id"]] = time

    return _build_result("FCFS", original, completion_times)


def sjf_scheduler(processes: List[Process]) -> Dict:
    procs = _clone_processes(processes)
    original = _clone_processes(processes)

    time = 0
    completed = 0
    n = len(procs)
    done = set()
    completion_times: Dict[int, int] = {}

    while completed < n:
        available = [p for p in procs if p["arrival_time"] <= time and p["id"] not in done]
        if not available:
            time += 1
            continue

        current = min(available, key=lambda x: (x["burst_time"], x["arrival_time"], x["id"]))
        time += current["burst_time"]
        completion_times[current["id"]] = time
        done.add(current["id"])
        completed += 1

    return _build_result("SJF", original, completion_times)


def round_robin_scheduler(processes: List[Process], quantum: int = 4) -> Dict:
    procs = _clone_processes(processes)
    original = _clone_processes(processes)

    time = 0
    completed = 0
    n = len(procs)
    arrival_index = 0
    ready_queue: List[Process] = []
    remaining = {p["id"]: p["remaining_time"] for p in procs}
    completion_times: Dict[int, int] = {}

    procs.sort(key=lambda x: (x["arrival_time"], x["id"]))

    while completed < n:
        while arrival_index < n and procs[arrival_index]["arrival_time"] <= time:
            ready_queue.append(procs[arrival_index])
            arrival_index += 1

        if not ready_queue:
            if arrival_index < n:
                time = max(time, procs[arrival_index]["arrival_time"])
                continue
            break

        current = ready_queue.pop(0)
        pid = current["id"]
        run_time = min(quantum, remaining[pid])

        remaining[pid] -= run_time
        time += run_time

        while arrival_index < n and procs[arrival_index]["arrival_time"] <= time:
            ready_queue.append(procs[arrival_index])
            arrival_index += 1

        if remaining[pid] > 0:
            ready_queue.append(current)
        else:
            completion_times[pid] = time
            completed += 1

    return _build_result("Round Robin", original, completion_times)