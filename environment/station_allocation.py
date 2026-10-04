"""A station's power split over its EVs by a fixed rule (EVMA_STATION_RULE_ALLOCATION).

The station's actor emits one scalar. Its sign selects charging or discharging,
and its magnitude is a fraction of the station's feasible total in that
direction. The rule splits that total using the station's own EVs only. A
charging total goes to the EVs with the least laxity first; a
discharging total is taken from the EVs with the most laxity first; no EV
charges while another discharges. Laxity is the steps left before departure
minus the steps full-rating charging needs to reach the target, so an EV above
its target has more laxity than its steps left. The split is piecewise linear
in the total, so a learner can differentiate through it.
"""

from __future__ import annotations

import torch


def laxity_steps(remaining_steps, need_soc_pct, capacity_kwh, max_power_kw, step_hours):
    """Steps to spare before departure at full-rating charging (need may be negative)."""
    full_step_pct = torch.clamp(max_power_kw * float(step_hours) * 100.0 / torch.clamp(capacity_kwh, min=1e-6),
                                min=1e-6)
    return remaining_steps - need_soc_pct / full_step_pct


def _fill(amount, capacity, order):
    """Give `amount` to slots in `order`, each up to its capacity."""
    cap_sorted = torch.gather(capacity, -1, order)
    before = torch.cumsum(cap_sorted, dim=-1) - cap_sorted
    got_sorted = torch.minimum(torch.clamp(amount.unsqueeze(-1) - before, min=0.0), cap_sorted)
    return torch.zeros_like(capacity).scatter(-1, order, got_sorted)


def allocate_by_laxity(total_kw, lo_kw, hi_kw, laxity, present):
    """Split each station's total over its EVs; tensors [..., n_ev], total_kw [...].

    lo_kw <= 0 <= hi_kw are each EV's feasible power this step. The result sums
    to total_kw clipped to [sum lo, sum hi] and respects every EV's bounds.
    """
    present = present.to(dtype=torch.bool)
    big = torch.finfo(laxity.dtype).max
    charge_cap = torch.where(present, torch.clamp(hi_kw, min=0.0), torch.zeros_like(hi_kw))
    discharge_cap = torch.where(present, torch.clamp(-lo_kw, min=0.0), torch.zeros_like(lo_kw))
    # Stable sorts keep ties in slot order; empty slots go last either way.
    charge_order = torch.argsort(torch.where(present, laxity, torch.full_like(laxity, big)), dim=-1, stable=True)
    discharge_order = torch.argsort(torch.where(present, -laxity, torch.full_like(laxity, big)), dim=-1, stable=True)
    charged = _fill(torch.clamp(total_kw, min=0.0), charge_cap, charge_order)
    discharged = _fill(torch.clamp(-total_kw, min=0.0), discharge_cap, discharge_order)
    return charged - discharged
