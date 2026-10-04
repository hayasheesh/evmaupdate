"""The departure force-charging floor, as a vectorized action bound.

training.system_controller.apply_force_charging raises an EV's action only
when the EV would otherwise no longer reach its departure target: once the
energy still needed (plus a slack) exceeds what the remaining steps after this
one can deliver at the EV's rated power. This module computes the same lowest
admissible action for many EVs at once, so the environment can apply it during
training and the learner can apply it to proposed actions. Every input is the
EV's own state, the same values the station observes.
"""

from __future__ import annotations

import torch

NO_FLOOR = -1.0


def force_floor_fraction(
    need_soc_pct: torch.Tensor,
    remaining_steps: torch.Tensor,
    capacity_kwh: torch.Tensor,
    max_power_kw: torch.Tensor,
    step_hours: float,
    slack_kwh: float = 0.0,
) -> torch.Tensor:
    """Lowest action, as a fraction of rated power in [-1, 1], that keeps the target reachable.

    Matches apply_force_charging: need_kwh includes the slack, so with a
    positive slack even an EV at its target is held to a small charge on its
    last step; no floor when that need is zero or the EV has no step left;
    otherwise floor_kwh = need_kwh - (remaining_steps - 1) * max_step_kwh, no
    floor while floor_kwh <= -max_step_kwh, and the floor is capped at full
    charging power.
    """

    max_step_kwh = torch.clamp(max_power_kw * float(step_hours), min=1e-9)
    need_kwh = torch.clamp(need_soc_pct, min=0.0) * capacity_kwh / 100.0
    need_with_slack = need_kwh + max(0.0, float(slack_kwh))
    floor = need_with_slack / max_step_kwh - (remaining_steps - 1.0)
    floor = torch.clamp(floor, NO_FLOOR, 1.0)
    unconstrained = (need_with_slack <= 0.0) | (remaining_steps <= 0.0)
    return torch.where(unconstrained, torch.full_like(floor, NO_FLOOR), floor)
