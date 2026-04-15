# algorithms/schedulers.py

import copy
from typing import List, Dict


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def _deep_copy_processes(processes: List[Dict]) -> List[Dict]:
    """Return a deep copy so original input is never mutated."""
    return copy.deepcopy(processes)


# ------------------------------------------------------------------
# 1. FCFS — First Come First Served (non-preemptive)
# ------------------------------------------------------------------

def fcfs(processes: List[Dict]) -> Dict:
    """
    First Come First Served scheduling.
    Processes are executed in order of arrival_time.
    Ties broken by process id.
    """
    procs = sorted(
        _deep_copy_processes(processes),
        key=lambda p: (p["arrival_time"], p["id"]),
    )

    current_time   = 0
    total_waiting  = 0.0
    total_turnaround = 0.0
    n = len(procs)

    for p in procs:
        # CPU may be idle if next process hasn't arrived yet
        if current_time < p["arrival_time"]:
            current_time = p["arrival_time"]

        waiting_time     = current_time - p["arrival_time"]
        current_time    += p["burst_time"]
        turnaround_time  = current_time - p["arrival_time"]

        total_waiting    += waiting_time
        total_turnaround += turnaround_time

    total_time = current_time  # time when last process finishes

    return {
        "avg_waiting_time"    : total_waiting    / n,
        "avg_turnaround_time" : total_turnaround / n,
        "throughput"          : n / total_time if total_time > 0 else 0.0,
    }


# ------------------------------------------------------------------
# 2. SJF — Shortest Job First (non-preemptive)
# ------------------------------------------------------------------

def sjf(processes: List[Dict]) -> Dict:
    """
    Shortest Job First scheduling (non-preemptive).
    At each scheduling decision, the arrived process with the
    shortest burst_time is chosen. Ties broken by arrival_time, then id.
    """
    procs        = _deep_copy_processes(processes)
    remaining    = list(procs)           # processes not yet started
    current_time = 0
    completed    = 0
    n            = len(procs)

    total_waiting    = 0.0
    total_turnaround = 0.0

    while completed < n:
        # Collect all processes that have arrived
        available = [p for p in remaining if p["arrival_time"] <= current_time]

        if not available:
            # CPU idle — jump to the next arrival
            next_arrival = min(p["arrival_time"] for p in remaining)
            current_time = next_arrival
            continue

        # Pick shortest burst; tie-break on arrival_time then id
        chosen = min(
            available,
            key=lambda p: (p["burst_time"], p["arrival_time"], p["id"]),
        )

        waiting_time     = current_time - chosen["arrival_time"]
        current_time    += chosen["burst_time"]
        turnaround_time  = current_time - chosen["arrival_time"]

        total_waiting    += waiting_time
        total_turnaround += turnaround_time

        remaining.remove(chosen)
        completed += 1

    total_time = current_time

    return {
        "avg_waiting_time"    : total_waiting    / n,
        "avg_turnaround_time" : total_turnaround / n,
        "throughput"          : n / total_time if total_time > 0 else 0.0,
    }


# ------------------------------------------------------------------
# 3. Round Robin (quantum = 4, preemptive)
# ------------------------------------------------------------------

def round_robin(processes: List[Dict], quantum: int = 4) -> Dict:
    """
    Round Robin scheduling with configurable time quantum (default = 4).
    Preemptive: each process runs for at most `quantum` units per turn.
    Processes are enqueued in arrival order; a process that is
    preempted re-enters at the back of the ready queue.
    """
    procs        = sorted(
        _deep_copy_processes(processes),
        key=lambda p: (p["arrival_time"], p["id"]),
    )

    n            = len(procs)
    current_time = 0
    queue        = []           # ready queue  (indices into procs)
    enqueued     = [False] * n  # track who has been added to queue
    pointer      = 0            # index into sorted procs for arrivals

    total_waiting    = 0.0
    total_turnaround = 0.0
    completed        = 0

    # Enqueue all processes that arrive at time 0
    while pointer < n and procs[pointer]["arrival_time"] <= current_time:
        queue.append(pointer)
        enqueued[pointer] = True
        pointer += 1

    while completed < n:
        if not queue:
            # CPU idle — advance to next arrival
            if pointer < n:
                current_time = procs[pointer]["arrival_time"]
                while pointer < n and procs[pointer]["arrival_time"] <= current_time:
                    queue.append(pointer)
                    enqueued[pointer] = True
                    pointer += 1
            continue

        idx     = queue.pop(0)
        process = procs[idx]

        # How long does this process run this slice?
        run_time     = min(quantum, process["remaining_time"])
        process["remaining_time"] -= run_time
        current_time += run_time

        # Enqueue any processes that arrived during this slice
        while pointer < n and procs[pointer]["arrival_time"] <= current_time:
            queue.append(pointer)
            enqueued[pointer] = True
            pointer += 1

        if process["remaining_time"] == 0:
            # Process finished
            turnaround_time  = current_time - process["arrival_time"]
            waiting_time     = turnaround_time - process["burst_time"]
            waiting_time     = max(waiting_time, 0)

            total_turnaround += turnaround_time
            total_waiting    += waiting_time
            completed        += 1
        else:
            # Preempted — re-enter at back of queue
            queue.append(idx)

    total_time = current_time

    return {
        "avg_waiting_time"    : total_waiting    / n,
        "avg_turnaround_time" : total_turnaround / n,
        "throughput"          : n / total_time if total_time > 0 else 0.0,
    }