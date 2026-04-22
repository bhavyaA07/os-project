# utils/process_generator.py

from __future__ import annotations

from typing import List, Dict, Optional
import random


Process = Dict[str, int]


def generate_processes(
    num_processes: int,
    max_arrival_time: int = 20,
    min_burst_time: int = 1,
    max_burst_time: int = 15,
    min_priority: int = 1,
    max_priority: int = 5,
    seed: Optional[int] = 42,
) -> List[Process]:
    rng = random.Random(seed)
    processes: List[Process] = []

    for i in range(num_processes):
        burst = rng.randint(min_burst_time, max_burst_time)
        processes.append(
            {
                "id": i,
                "arrival_time": rng.randint(0, max_arrival_time),
                "burst_time": burst,
                "remaining_time": burst,
                "priority": rng.randint(min_priority, max_priority),
            }
        )

    processes.sort(key=lambda p: (p["arrival_time"], p["id"]))
    return processes


def clone_processes(processes: List[Process]) -> List[Process]:
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


def generate_workload_bank(
    num_datasets: int = 200,
    min_processes: int = 10,
    max_processes: int = 35,
    max_arrival_time: int = 60,
    min_burst_time: int = 1,
    max_burst_time: int = 20,
    min_priority: int = 1,
    max_priority: int = 5,
    seed: int = 42,
) -> List[List[Process]]:
    rng = random.Random(seed)
    workloads: List[List[Process]] = []

    for _ in range(num_datasets):
        n = rng.randint(min_processes, max_processes)
        ds_seed = rng.randint(0, 10**9)
        workloads.append(
            generate_processes(
                num_processes=n,
                max_arrival_time=max_arrival_time,
                min_burst_time=min_burst_time,
                max_burst_time=max_burst_time,
                min_priority=min_priority,
                max_priority=max_priority,
                seed=ds_seed,
            )
        )

    return workloads