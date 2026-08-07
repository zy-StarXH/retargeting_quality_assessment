"""Boundary-safe multi-motion command for the pinned whole_body_tracking task.

This file deliberately lives in the project integration layer.  It does not
modify the user's external whole_body_tracking checkout.
"""

from __future__ import annotations

import numpy as np
import torch

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand


class BoundaryAwareMotionCommand(MotionCommand):
    """Sample each source motion uniformly and never step across a boundary."""

    def __init__(self, cfg, env):
        with np.load(cfg.motion_file, allow_pickle=False) as data:
            if "motion_starts" not in data or "motion_ends" not in data:
                raise ValueError("Step-11 bundles require motion_starts and motion_ends")
            self._starts_numpy = np.asarray(data["motion_starts"], dtype=np.int64)
            self._ends_numpy = np.asarray(data["motion_ends"], dtype=np.int64)
            self.motion_names = tuple(str(value) for value in np.asarray(data["motion_names"]).tolist())
        super().__init__(cfg, env)
        self.motion_starts = torch.as_tensor(self._starts_numpy, dtype=torch.long, device=self.device)
        self.motion_ends = torch.as_tensor(self._ends_numpy, dtype=torch.long, device=self.device)
        self.current_motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.current_motion_end_steps = torch.full(
            (self.num_envs,), int(self.motion_ends[0]), dtype=torch.long, device=self.device
        )

    def _adaptive_sampling(self, env_ids):
        """Controlled sampler: motion-uniform, then time-uniform.

        Failure-adaptive sampling is intentionally disabled because it would
        allocate more optimizer experience to lower-quality cohorts and would
        confound the intended equal-budget comparison.
        """
        count = len(env_ids)
        if count == 0:
            return
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        motion_ids = torch.randint(0, len(self.motion_starts), (count,), device=self.device)
        starts = self.motion_starts[motion_ids]
        ends = self.motion_ends[motion_ids]
        # A reference must remain in the same source motion for a complete
        # episode.  Frames near the end are still traversed, but never selected
        # as a start if fewer than one episode remains.
        horizon = int(getattr(self._env, "max_episode_length", 1))
        start_span = (ends - starts - horizon).clamp_min(1)
        offsets = (torch.rand(count, device=self.device) * start_span).long()
        self.time_steps[ids] = starts + offsets
        self.current_motion_ids[ids] = motion_ids
        self.current_motion_end_steps[ids] = ends
        # Preserve the upstream logging contract while clearly representing a
        # uniform sampler rather than failure-adaptive bins.
        self.metrics["sampling_entropy"][:] = 1.0
        self.metrics["sampling_top1_prob"][:] = 1.0 / float(len(self.motion_starts))
        self.metrics["sampling_top1_bin"][:] = 0.0

    def _update_command(self):
        # The parent increments once, then resamples only at global EOF.  Map
        # per-motion EOF to global EOF immediately before that increment.
        crossing = self.time_steps >= (self.current_motion_end_steps - 1)
        self.time_steps[crossing] = self.motion.time_step_total - 1
        super()._update_command()
