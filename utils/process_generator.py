# utils/process_generator.py

import numpy as np
from typing import List, Dict


def generate_processes(
    num_processes: int = 10,
    max_arrival_time: int = 20,
    max_burst_time: int = 15,
    max_priority: int = 5,
    seed: int = 42
) -> List[Dict]:
    """
    Generate synthetic CPU processes using numpy.

    Returns a list of process dicts sorted by arrival_time.
    Each dict follows the strict schema:
        {
            'id': int,
            'arrival_time': int,
            'burst_time': int,
            'remaining_time': int,
            'priority': int
        }
    """
    rng = np.random.default_rng(seed)

    arrival_times = np.sort(
        rng.integers(0, max_arrival_time + 1, size=num_processes)
    ).tolist()

    burst_times = rng.integers(1, max_burst_time + 1, size=num_processes).tolist()
    priorities  = rng.integers(1, max_priority  + 1, size=num_processes).tolist()

    processes: List[Dict] = []
    for i in range(num_processes):
        processes.append({
            'id'            : i + 1,
            'arrival_time'  : int(arrival_times[i]),
            'burst_time'    : int(burst_times[i]),
            'remaining_time': int(burst_times[i]),   # initialised = burst_time
            'priority'      : int(priorities[i]),
        })

    return processes