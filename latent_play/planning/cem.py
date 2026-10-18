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
        prior_correction: float = 0.5,
    ) -> None:
        self.dynamics = dynamics.eval()
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.iterations = iterations
        self.temperature = temperature
        self.alignment = alignment
        self.prior_correction = prior_correction
        self.num_actions = dynamics.cfg.num_actions

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
        if z_curr.ndim == 3:  # [N, D] -> [1, N, D]
            z_curr = z_curr.unsqueeze(0)
        if z_goal.ndim == 3:
            z_goal = z_goal.unsqueeze(0)

        H = self.horizon
        M = self.num_samples
        A = self.num_actions

        # Initialize uniform logits over H timesteps: [H, A]
        logits = torch.zeros(H, A, device=device)

        # Expand z_curr for parallel rollout: [M, 1, ...]
        z_init = z_curr.repeat(M, 1, 1, 1) if z_curr.ndim == 4 else z_curr.repeat(M, 1, 1)

        best_seq = None
        best_cost = float("inf")

        for it in range(self.iterations):
            # Sample M sequences: [M, H]
            probs = F.softmax(logits / self.temperature, dim=-1)  # [H, A]
            dist = torch.distributions.Categorical(probs)
            actions = dist.sample((M,))  # [M, H]

            # Autoregressive rollout in latent space: [M, H, ...]
            traj = self.dynamics.rollout(z_init, actions)
            z_final = traj[:, -1]  # [M, ...]

            # Cost: combination of terminal distance and closest approach along the horizon
            # This ensures goals closer than H steps are reached immediately without overshooting.
            step_dists = (traj.float() - z_goal.unsqueeze(1).float()).abs().flatten(2).mean(dim=2)  # [M, H]
            min_dist = step_dists.min(dim=1).values
            final_dist = (z_final.float() - z_goal.float()).abs().flatten(1).mean(dim=1)
            costs = 0.5 * final_dist + 0.5 * min_dist

            # Find elites (lowest cost)
            elite_idx = costs.argsort()[: self.num_elites]
            elite_actions = actions[elite_idx]  # [num_elites, H]

            if costs[elite_idx[0]].item() < best_cost:
                best_cost = costs[elite_idx[0]].item()
                best_seq = elite_actions[0]

            # Update distribution parameters from elites
            # One-hot encode elite actions: [num_elites, H, A]
            one_hot = F.one_hot(elite_actions, num_classes=A).float()
            # Mean frequency across elites: [H, A]
            elite_probs = one_hot.mean(dim=0).clamp_min(1e-6)
            logits = torch.log(elite_probs)

        assert best_seq is not None
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
