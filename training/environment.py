# training/environment.py

import copy
from typing import List, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CPUSchedulingEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self, process_list: List[Dict], max_processes: int = 10):
        super().__init__()

        if not process_list:
            raise ValueError("process_list must not be empty.")

        self.original_process_list: List[Dict] = process_list
        self.max_processes: int = max_processes

        self.action_space = spaces.Discrete(max_processes)

        obs_size = max_processes * 4
        self.observation_space = spaces.Box(
            low=np.zeros(obs_size, dtype=np.float32),
            high=np.full(obs_size, 1e6, dtype=np.float32),
            dtype=np.float32,
        )

        self.current_time: int = 0
        self.ready_queue: List[Dict] = []
        self.process_list: List[Dict] = []
        self.completed_processes: List[Dict] = []
        self.waiting_times: Dict[int, float] = {}

    def _arrive_processes(self) -> None:
        still_waiting = []
        for p in self.process_list:
            if p["arrival_time"] <= self.current_time:
                self.ready_queue.append(p)
                if p["id"] not in self.waiting_times:
                    self.waiting_times[p["id"]] = 0.0
            else:
                still_waiting.append(p)
        self.process_list = still_waiting

    def _build_observation(self) -> np.ndarray:
        obs = np.zeros(self.max_processes * 4, dtype=np.float32)
        for i, p in enumerate(self.ready_queue[: self.max_processes]):
            base = i * 4
            obs[base + 0] = np.float32(p["burst_time"])
            obs[base + 1] = np.float32(p["remaining_time"])
            obs[base + 2] = np.float32(self.waiting_times.get(p["id"], 0.0))
            obs[base + 3] = np.float32(p["priority"])
        return obs

    def _update_waiting_times(self, scheduled_id: Optional[int]) -> None:
        for p in self.ready_queue:
            if p["id"] != scheduled_id:
                self.waiting_times[p["id"]] = (
                    self.waiting_times.get(p["id"], 0.0) + 1.0
                )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self.process_list = copy.deepcopy(self.original_process_list)
        self.ready_queue = []
        self.completed_processes = []
        self.waiting_times = {}
        self.current_time = 0

        self._arrive_processes()

        return self._build_observation(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        reward = 0.0

        self._arrive_processes()

        if not self.ready_queue:
            if not self.process_list:
                obs = self._build_observation()
                return obs, np.float32(-1.0), True, False, {}
            self.current_time += 1
            self._arrive_processes()
            reward -= 2.0
            obs = self._build_observation()
            terminated = not self.ready_queue and not self.process_list
            return obs, np.float32(reward), bool(terminated), False, {}

        action = int(action)
        if action >= len(self.ready_queue):
            action = 0
            reward -= 0.5

        process = self.ready_queue[action]
        scheduled_id = process["id"]

        self._update_waiting_times(scheduled_id)

        process["remaining_time"] -= 1
        self.current_time += 1

        total_waiting = sum(self.waiting_times.values())
        reward -= 0.01 * total_waiting

        if process["remaining_time"] <= 0:
            turnaround = self.current_time - process["arrival_time"]
            waiting    = max(turnaround - process["burst_time"], 0.0)

            process["turnaround_time"] = turnaround
            process["waiting_time"]    = waiting

            self.completed_processes.append(process)
            self.ready_queue.pop(action)

            reward += max(10.0 - 0.1 * turnaround, 1.0)

        terminated = (
            len(self.completed_processes) == len(self.original_process_list)
        )

        obs = self._build_observation()
        return obs, np.float32(reward), bool(terminated), False, {}

    def render(self):
        pass