r"""Latent space Cross-Entropy Method (CEM) / MPC Planner (V-JEPA 2-AC style).

Plans goal-directed trajectories purely in representation space:
1. Encode current observation x_t -> z_t and goal observation x_goal -> z_goal
2. Sample M candidate action sequences of horizon H from categorical distributions
3. Roll out candidates in latent space using JEPADynamics: \hat{z}_{t+1..t+H}
4. Score trajectories by representational proximity to goal: - || \hat{z}_{final} - z_{goal} ||_1
5. Select elite sequences, update categorical parameters, iterate
6. Execute the first action a_0 in the environment (receding-horizon MPC)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from latent_play.models import JEPADynamics
from latent_play.planning.alignment import ActionAlignment


class CEMPlanner:
    def __init__(
        self,
        dynamics: JEPADynamics,
        horizon: int = 8,
        num_samples: int = 128,
        num_elites: int = 16,
        iterations: int = 4,
        temperature: float = 1.0,
        alignment: ActionAlignment | None = None,
        prior_correction: float = 0.0,
        alpha: float = 0.5,
        stagnation_penalty: float = 0.0,
        stagnation_threshold: float = 0.01,
        time_to_goal_weight: float = 0.02,
        noop_penalty: float = 0.05,
        cost_metric: str = "auto",
    ) -> None:
        self.dynamics = dynamics.eval()
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.iterations = iterations
        self.temperature = temperature
        self.alignment = alignment
        self.prior_correction = prior_correction
        self.alpha = alpha
        self.stagnation_penalty = stagnation_penalty
        self.stagnation_threshold = stagnation_threshold
        self.time_to_goal_weight = time_to_goal_weight
        self.noop_penalty = noop_penalty
        self.cost_metric = cost_metric
        self.num_actions = dynamics.cfg.num_actions
        self.prev_logits: torch.Tensor | None = None

        if self.alignment is not None:
            assert self.num_actions == self.alignment.num_codes, (
                f"Dynamics model num_actions ({self.num_actions}) must match alignment num_codes ({self.alignment.num_codes})"
            )

    def reset(self) -> None:
        """Clears warm-started plan distribution at the start of a new episode."""
        self.prev_logits = None

    @torch.no_grad()
    def plan(
        self,
        z_curr: torch.Tensor,
        z_goal: torch.Tensor,
    ) -> tuple[int, torch.Tensor]:
        """Plans action sequence to reach z_goal from z_curr.

        Args:
            z_curr: [1, ..., D] or [..., D] current state representation
            z_goal: [1, ..., D] or [..., D] goal state representation

        Returns:
            first_action: int real executable environment action
            best_sequence: [H] tensor of planned action sequence
        """
        device = z_curr.device
        pool_tokens = getattr(self.dynamics.cfg, "pool_tokens", False)
        if pool_tokens:
            if z_curr.ndim == 1:
                z_curr = z_curr.unsqueeze(0)  # [1, D]
            if z_goal.ndim == 1:
                z_goal = z_goal.unsqueeze(0)  # [1, D]
        else:
            if z_curr.ndim == 2:
                z_curr = z_curr.unsqueeze(0)  # [1, N, D]
            if z_goal.ndim == 2:
                z_goal = z_goal.unsqueeze(0)  # [1, N, D]

        H = self.horizon
        M = self.num_samples
        A = self.num_actions

        # Warm-start logits from shifted previous plan if available; otherwise uniform
        if self.prev_logits is not None and self.prev_logits.shape == (H, A):
            logits = torch.zeros(H, A, device=device)
            logits[:-1] = self.prev_logits[1:]
        else:
            logits = torch.zeros(H, A, device=device)

        # Expand z_curr for parallel rollout: [M, 1, ...]
        z_init = z_curr.repeat(M, *([1] * (z_curr.ndim - 1)))
        z_goal_target = z_goal.squeeze(0)  # [D] or [N, D]

        # Precompute spatial smoothing kernel if evaluating on 2D patch grid
        G = int(self.dynamics.cfg.image_size // self.dynamics.cfg.patch)
        effective_metric = self.cost_metric
        if effective_metric == "auto":
            if (not pool_tokens) and (z_goal_target.ndim == 2) and (z_goal_target.shape[0] == G * G):
                effective_metric = "spatial_conv"
            else:
                effective_metric = "l1"

        if effective_metric == "spatial_conv":
            D = self.dynamics.cfg.dim
            kernel_size = 5
            sigma = 1.5
            coords = torch.arange(kernel_size, device=device) - kernel_size // 2
            g1d = torch.exp(-coords**2 / (2 * sigma**2))
            g2d = (g1d[:, None] * g1d[None, :])
            kernel = (g2d / g2d.sum()).unsqueeze(0).unsqueeze(0).repeat(D, 1, 1, 1)

            goal_2d = z_goal_target.reshape(1, G, G, D).permute(0, 3, 1, 2)
            s_goal = F.conv2d(goal_2d, kernel, padding=kernel_size // 2, groups=D)

        best_seq = None
        best_cost = float("inf")

        for it in range(self.iterations):
            # Sample M sequences: [M, H]
            probs = F.softmax(logits / self.temperature, dim=-1)  # [H, A]
            dist = torch.distributions.Categorical(probs)
            actions = dist.sample((M,))  # [M, H]

            # Autoregressive rollout in latent space: [M, H, ...]
            traj = self.dynamics.rollout(z_init, actions)

            # Cost: combination of closest approach along the horizon and early arrival incentive.
            # This ensures goals closer than H steps are reached immediately without being penalized
            # for subsequent actions taken after arrival.
            if effective_metric == "spatial_conv":
                traj_2d = traj.reshape(M * H, G, G, D).permute(0, 3, 1, 2)
                s_traj = F.conv2d(traj_2d, kernel, padding=kernel_size // 2, groups=D)
                step_dists = (s_traj - s_goal).abs().reshape(M, H, -1).mean(dim=-1)
            elif effective_metric == "cosine":
                sim = F.cosine_similarity(traj.float(), z_goal_target.float(), dim=-1)
                step_dists = (1.0 - sim).reshape(M, H, -1).mean(dim=-1)
            elif effective_metric == "pooled":
                t_p = traj.float().mean(dim=2) if traj.ndim == 4 else traj.float()
                g_p = z_goal_target.float().mean(dim=0) if z_goal_target.ndim == 2 else z_goal_target.float()
                step_dists = (t_p - g_p).abs().mean(dim=-1)
            else:  # "l1"
                diff = traj.float() - z_goal_target.float()
                step_dists = diff.abs().reshape(M, H, -1).mean(dim=-1)  # [M, H]

            min_vals, min_steps = step_dists.min(dim=1)
            costs = min_vals + self.time_to_goal_weight * min_steps.float()

            # Penalize picking NOOP as first action when attempting to navigate to goal
            if self.noop_penalty > 0.0:
                first_actions = actions[:, 0]
                if self.alignment is not None:
                    z_ctx = z_curr.repeat(M, *([1] * (z_curr.ndim - 1))) if self.alignment.use_state_context else None
                    first_real = self.alignment.predict_action(first_actions, z=z_ctx, prior_correction=self.prior_correction)
                    is_noop = (first_real == 0).float()
                else:
                    is_noop = (first_actions == 0).float()
                costs = costs + self.noop_penalty * is_noop

            # Stagnation penalty: penalize plans that get stuck against obstacles or produce near-zero displacement
            if self.stagnation_penalty > 0.0 and traj.shape[1] > 1:
                displacement = (traj[:, 1:].float() - traj[:, :-1].float()).abs().reshape(M, H - 1, -1).mean(dim=-1)  # [M, H-1]
                stuck = (displacement < self.stagnation_threshold).float().mean(dim=1)  # [M]
                costs = costs + self.stagnation_penalty * stuck

            # Find elites (lowest cost)
            elite_idx = costs.argsort()[: self.num_elites]
            elite_actions = actions[elite_idx]  # [num_elites, H]

            if costs[elite_idx[0]].item() <= best_cost:
                best_cost = costs[elite_idx[0]].item()
                best_seq = elite_actions[0]

            # Update distribution parameters from elites with momentum smoothing in probability space
            one_hot = F.one_hot(elite_actions, num_classes=A).float()
            elite_probs = one_hot.mean(dim=0)  # [H, A]
            curr_probs = F.softmax(logits, dim=-1)
            smoothed_probs = (1.0 - self.alpha) * curr_probs + self.alpha * elite_probs
            logits = torch.log(smoothed_probs.clamp_min(1e-6))

        assert best_seq is not None
        self.prev_logits = logits.detach()
        first_action_id = int(best_seq[0].item())

        # If planning in latent action space, translate first action to real environment action
        if self.alignment is not None:
            code_t = torch.tensor([first_action_id], device=device)
            z_ctx = z_curr if self.alignment.use_state_context else None
            real_action = int(
                self.alignment.predict_action(
                    code_t, z=z_ctx, prior_correction=self.prior_correction
                ).item()
            )
            return real_action, best_seq

        return first_action_id, best_seq
