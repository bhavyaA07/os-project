# training/environment.py

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces


Process = Dict[str, int]


class CPUSchedulingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        process_list: List[Process],
        max_processes: int = 10,
        quantum_choices: Optional[List[int]] = None,
        starvation_threshold: int = 25,
        max_episode_steps: int = 8000,
    ):
        super().__init__()
        self.base_processes = self._clone_processes(process_list)
        self.max_processes = max_processes
        self.quantum_choices = quantum_choices or [1, 2, 4, 8]
        self.num_quantums = len(self.quantum_choices)
        self.starvation_threshold = starvation_threshold
        self.max_episode_steps = max_episode_steps

        self.action_space = spaces.Discrete(self.max_processes * self.num_quantums)

        obs_len = self.max_processes * 5 + 8
        self.observation_space = spaces.Box(
            low=0.0,
            high=1e6,
            shape=(obs_len,),
            dtype=np.float32,
        )

        self.processes: List[Process] = []
        self.ready_queue: List[Process] = []
        self.completed_processes: List[Dict] = []
        self.current_time = 0
        self.context_switches = 0
        self.last_pid: Optional[int] = None
        self.steps_taken = 0

    @staticmethod
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

    def _decode_action(self, action: int) -> Tuple[int, int]:
        slot_index = action // self.num_quantums
        quantum_index = action % self.num_quantums
        return slot_index, self.quantum_choices[quantum_index]

    def _get_waiting_time(self, p: Process) -> int:
        executed = p["burst_time"] - p["remaining_time"]
        return max(0, self.current_time - p["arrival_time"] - executed)

    def _refresh_ready_queue(self) -> None:
        self.ready_queue = [
            p for p in self.processes
            if p["arrival_time"] <= self.current_time and p["remaining_time"] > 0
        ]
        self.ready_queue.sort(key=lambda x: (x["arrival_time"], x["id"]))

    def _all_done(self) -> bool:
        return all(p["remaining_time"] == 0 for p in self.processes)

    def _completion_ratio(self) -> float:
        return len(self.completed_processes) / len(self.processes) if self.processes else 1.0

    def _get_observation(self) -> np.ndarray:
        self._refresh_ready_queue()
        obs = np.zeros((self.max_processes, 5), dtype=np.float32)

        visible = self.ready_queue[: self.max_processes]
        for i, p in enumerate(visible):
            obs[i, 0] = float(p["burst_time"])
            obs[i, 1] = float(self._get_waiting_time(p))
            obs[i, 2] = float(p["remaining_time"])
            obs[i, 3] = float(p["priority"])
            obs[i, 4] = 1.0

        waits = [self._get_waiting_time(p) for p in self.ready_queue] if self.ready_queue else [0]
        rems = [p["remaining_time"] for p in self.ready_queue] if self.ready_queue else [0]

        global_features = np.array(
            [
                float(self.current_time),
                float(len(self.ready_queue)),
                float(np.mean(waits)),
                float(np.max(waits)),
                float(np.min(rems)),
                float(np.sum(rems)),
                float(self.context_switches),
                float(self._completion_ratio()),
            ],
            dtype=np.float32,
        )

        return np.concatenate([obs.flatten(), global_features]).astype(np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.processes = self._clone_processes(self.base_processes)
        self.ready_queue = []
        self.completed_processes = []
        self.current_time = 0
        self.context_switches = 0
        self.last_pid = None
        self.steps_taken = 0
        return self._get_observation(), {}

    def step(self, action: int):
        self.steps_taken += 1
        reward = 0.0
        terminated = False
        truncated = False

        self._refresh_ready_queue()

        if self._all_done():
            terminated = True
            return self._get_observation(), 0.0, terminated, truncated, {}

        if self.steps_taken >= self.max_episode_steps:
            truncated = True
            return self._get_observation(), -2.0, terminated, truncated, {}

        if not self.ready_queue:
            self.current_time += 1
            reward -= 0.15
            self._refresh_ready_queue()
            if self._all_done():
                terminated = True
                reward += 15.0
            return self._get_observation(), float(reward), terminated, truncated, {}

        slot_index, quantum = self._decode_action(action)

        if slot_index >= len(self.ready_queue):
            self.current_time += 1
            reward -= 0.75
            self._refresh_ready_queue()
            return self._get_observation(), float(reward), terminated, truncated, {}

        current = self.ready_queue[slot_index]

        if self.last_pid is not None and self.last_pid != current["id"]:
            self.context_switches += 1
            reward -= 0.03

        self.last_pid = current["id"]

        before_waits = [self._get_waiting_time(p) for p in self.ready_queue]
        before_avg_wait = float(np.mean(before_waits)) if before_waits else 0.0
        before_max_wait = float(np.max(before_waits)) if before_waits else 0.0

        run_time = min(quantum, current["remaining_time"])
        used_cpu = 0
        completed_now = False

        for _ in range(run_time):
            current["remaining_time"] -= 1
            self.current_time += 1
            used_cpu += 1

            if current["remaining_time"] == 0:
                completion_time = self.current_time
                turnaround_time = completion_time - current["arrival_time"]
                waiting_time = turnaround_time - current["burst_time"]

                self.completed_processes.append(
                    {
                        "id": current["id"],
                        "arrival_time": current["arrival_time"],
                        "burst_time": current["burst_time"],
                        "remaining_time": 0,
                        "priority": current["priority"],
                        "completion_time": completion_time,
                        "turnaround_time": turnaround_time,
                        "waiting_time": waiting_time,
                    }
                )
                completed_now = True
                break

        self._refresh_ready_queue()

        after_waits = [self._get_waiting_time(p) for p in self.ready_queue]
        after_avg_wait = float(np.mean(after_waits)) if after_waits else 0.0
        after_max_wait = float(np.max(after_waits)) if after_waits else 0.0
        starvation_count = sum(1 for w in after_waits if w >= self.starvation_threshold)

        reward += 0.5 * used_cpu
        reward -= 0.01 * after_avg_wait
        reward -= 0.008 * after_max_wait
        reward -= 0.05 * starvation_count
        reward -= 0.02 * max(0.0, after_avg_wait - before_avg_wait)

        if completed_now:
            reward += 10.0

        if after_max_wait < before_max_wait:
            reward += 0.25

        if self._all_done():
            terminated = True
            reward += 20.0

        return self._get_observation(), float(reward), terminated, truncated, {}

    def render(self):
        visible = [p["id"] for p in self.ready_queue[: self.max_processes]]
        print(
            f"Time={self.current_time}, Ready={visible}, "
            f"Completed={len(self.completed_processes)}/{len(self.processes)}, "
            f"ContextSwitches={self.context_switches}, Steps={self.steps_taken}"
        )