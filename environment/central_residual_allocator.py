"""Central-observation rule baseline with a station-only public interface.

The physical controller is deliberately split into three boundaries:

1. Each station privately derives per-EV feasible powers and publishes only an
   aggregate flexibility envelope.
2. The central allocator receives those station aggregates and selects one
   target power per station.  It has no ``env`` argument and no per-EV tensor.
3. Each station privately maps its aggregate target back to its EVs.

This controller is retained as a comparison bound. It is not part of the
proposed distributed MARL system because it assumes that every station accepts
the power target selected by the central allocator.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from EnvConfig import POWER_TO_ENERGY


def capped_weighted_waterfill(
    headroom: torch.Tensor,
    weights: torch.Tensor,
    amount_kw: float,
    *,
    iterations: int = 16,
) -> torch.Tensor:
    """Allocate one scalar amount without sorting or exceeding headroom."""
    headroom = torch.clamp(headroom, min=0.0)
    if float(amount_kw) <= 0.0 or headroom.numel() == 0:
        return torch.zeros_like(headroom)

    valid = headroom > 1e-8
    weights = torch.where(
        valid, torch.clamp(weights, min=1e-6), torch.zeros_like(weights)
    )
    target = torch.minimum(
        headroom.sum(),
        torch.as_tensor(
            float(amount_kw), dtype=headroom.dtype, device=headroom.device
        ),
    )
    # The water level only has to reach the largest ratio among entries that
    # actually have headroom. Taking the maximum over all entries let a single
    # zero-headroom row with the 1e-6 weight floor push the upper bound up by
    # six orders of magnitude, which spent the bisection budget on an interval
    # the answer was never in.
    ratio = torch.where(
        valid,
        headroom / torch.clamp(weights, min=1e-6),
        torch.zeros_like(headroom),
    )
    lo = torch.zeros((), dtype=headroom.dtype, device=headroom.device)
    hi = torch.where(valid, ratio, torch.zeros_like(ratio)).max()
    for _ in range(max(int(iterations), 1)):
        mid = (lo + hi) * 0.5
        allocated = torch.minimum(headroom, mid * weights).sum()
        below = allocated < target
        lo = torch.where(below, mid, lo)
        hi = torch.where(below, hi, mid)

    allocation = torch.minimum(headroom, hi * weights)
    allocation = allocation * torch.minimum(
        torch.ones_like(target),
        target / torch.clamp(allocation.sum(), min=1e-8),
    )
    # Priority-proportional top-ups respect the tier ordering but cannot
    # guarantee the amount is placed: a row with a high weight and little spare
    # has its share clipped, and that clipped part was simply dropped. Two such
    # rounds left headroom unused while the request went unmet.
    for _ in range(2):
        remaining = torch.clamp(target - allocation.sum(), min=0.0)
        spare = torch.clamp(headroom - allocation, min=0.0)
        active_weights = torch.where(
            spare > 1e-8, weights, torch.zeros_like(weights)
        )
        share = (
            remaining
            * active_weights
            / torch.clamp(active_weights.sum(), min=1e-8)
        )
        allocation = allocation + torch.minimum(spare, share)
    # Spare-proportional, so each row receives at most its own spare and the
    # sum is exactly the remainder. `target` never exceeds total headroom, so
    # this closes the gap instead of leaving capacity on the table. The station
    # level has always finished this way; the central level did not.
    remaining = torch.clamp(target - allocation.sum(), min=0.0)
    spare = torch.clamp(headroom - allocation, min=0.0)
    allocation = allocation + (
        remaining * spare / torch.clamp(spare.sum(), min=1e-8)
    )
    return torch.minimum(allocation, headroom)


def _batched_capped_weighted_waterfill(
    headroom: torch.Tensor,
    weights: torch.Tensor,
    amounts_kw: torch.Tensor,
    *,
    iterations: int,
) -> torch.Tensor:
    """Run independent capped weighted fills for every station row.

    This is only a simulator batching optimization.  No value in one station
    row affects another row, so the operation can be deployed independently at
    each station controller.  A few proportional saturation passes preserve
    the priority ordering; the final spare-proportional pass fills the exact
    requested amount without the 16 GPU kernel launches needed by bisection.
    With at most ten EVs per station that distinction materially affects
    training speed but not feasibility or the aggregate station target.
    """
    if headroom.dim() != 2:
        raise ValueError("batched station headroom must be rank 2")
    if weights.shape != headroom.shape:
        raise ValueError("batched station weights must match headroom")
    if amounts_kw.shape != headroom.shape[:1]:
        raise ValueError("one station allocation amount is required per row")

    headroom = torch.clamp(headroom, min=0.0)
    amounts_kw = torch.clamp(amounts_kw, min=0.0)
    weights = torch.where(
        headroom > 1e-8,
        torch.clamp(weights, min=1e-6),
        torch.zeros_like(weights),
    )
    target = torch.minimum(headroom.sum(dim=1), amounts_kw)
    allocation = torch.zeros_like(headroom)
    # Three passes resolve the meaningful priority tiers for ten-slot
    # stations. ``iterations`` remains in the API because the fleet-level
    # central water-fill still uses the configured bisection count.
    for _ in range(3):
        remaining = torch.clamp(target - allocation.sum(dim=1), min=0.0)
        spare = torch.clamp(headroom - allocation, min=0.0)
        active_weights = torch.where(
            spare > 1e-8, weights, torch.zeros_like(weights)
        )
        share = (
            remaining[:, None]
            * active_weights
            / torch.clamp(active_weights.sum(dim=1), min=1e-8)[:, None]
        )
        allocation = allocation + torch.minimum(spare, share)

    # The remaining amount cannot exceed total spare.  A spare-proportional
    # final pass is therefore exactly capped and removes all numerical residue.
    remaining = torch.clamp(target - allocation.sum(dim=1), min=0.0)
    spare = torch.clamp(headroom - allocation, min=0.0)
    allocation = allocation + (
        remaining[:, None]
        * spare
        / torch.clamp(spare.sum(dim=1), min=1e-8)[:, None]
    )
    return torch.minimum(allocation, headroom)


def _aggregate_priority(
    headroom: torch.Tensor, per_ev_priority: torch.Tensor
) -> torch.Tensor:
    """Compress private per-EV priorities into one station-level scalar."""
    total = torch.clamp(headroom, min=0.0).sum(dim=1)
    weighted = (
        torch.clamp(headroom, min=0.0)
        * torch.clamp(per_ev_priority, min=0.0)
    ).sum(dim=1)
    return torch.where(
        total > 1e-8,
        weighted / torch.clamp(total, min=1e-8),
        torch.zeros_like(total),
    )


@dataclass(frozen=True)
class StationFlexibilityEnvelope:
    """The complete public interface from stations to the central allocator.

    Every field is one scalar per station.  There are intentionally no EV-slot
    dimensions, SoCs, capacities, deadlines, IDs, or masks in this object.
    """

    raw_power_kw: torch.Tensor
    safe_min_power_kw: torch.Tensor
    safe_max_power_kw: torch.Tensor
    toward_target_charge_headroom_kw: torch.Tensor
    surplus_charge_headroom_kw: torch.Tensor
    remove_charge_headroom_kw: torch.Tensor
    safe_discharge_headroom_kw: torch.Tensor
    toward_target_priority: torch.Tensor
    surplus_charge_priority: torch.Tensor
    remove_charge_priority: torch.Tensor
    safe_discharge_priority: torch.Tensor

    @property
    def num_stations(self) -> int:
        return int(self.raw_power_kw.numel())

    def validate(self) -> None:
        expected = self.raw_power_kw.shape
        if len(expected) != 1:
            raise ValueError("station envelope fields must be one-dimensional")
        for name, value in self.__dict__.items():
            if value.shape != expected:
                raise ValueError(
                    f"station envelope {name} has shape {tuple(value.shape)}, "
                    f"expected {tuple(expected)}"
                )
        if torch.any(self.safe_min_power_kw > self.raw_power_kw + 1e-5):
            raise ValueError("station safe minimum exceeds raw station power")
        if torch.any(self.safe_max_power_kw < self.raw_power_kw - 1e-5):
            raise ValueError("station safe maximum is below raw station power")


@dataclass
class _StationLocalDispatchContext:
    """Private station-edge data; this object never enters central allocation."""

    actor_actions: torch.Tensor
    present: torch.Tensor
    max_power_kw: torch.Tensor
    raw_power_kw: torch.Tensor
    physical_upper_kw: torch.Tensor
    target_upper_kw: torch.Tensor
    correction_lower_kw: torch.Tensor
    toward_target_priority: torch.Tensor
    flexible_priority: torch.Tensor
    surplus_charge_priority: torch.Tensor
    envelope: StationFlexibilityEnvelope


def build_station_flexibility_envelope(
    env,
    actor_actions: torch.Tensor,
    *,
    departure_slack_kwh: float = 0.1,
) -> _StationLocalDispatchContext:
    """Build private station state and its public aggregate envelope."""
    actions = torch.clamp(actor_actions, -1.0, 1.0)
    shape = (int(env.num_stations), int(env.max_ev_per_station))
    dtype = actions.dtype
    dev = actions.device
    present = torch.zeros(shape, dtype=torch.bool, device=dev)
    capacity = torch.ones(shape, dtype=dtype, device=dev)
    max_power = torch.zeros(shape, dtype=dtype, device=dev)
    soc_pct = torch.zeros(shape, dtype=dtype, device=dev)
    target_pct = torch.zeros(shape, dtype=dtype, device=dev)
    remaining = torch.ones(shape, dtype=dtype, device=dev)

    # This is station-edge collection.  Each row uses only its own local EVs.
    for station in range(int(env.num_stations)):
        ordered = env._get_sorted_active_evs(station)
        count = int(ordered.numel())
        if count <= 0:
            continue
        present[station, :count] = True
        capacity[station, :count] = env.ev_capacity_kwh[station, ordered]
        max_power[station, :count] = env.ev_max_power_kw[station, ordered]
        soc_pct[station, :count] = env.soc[station, ordered]
        target_pct[station, :count] = env.target[station, ordered]
        remaining[station, :count] = env._remaining_action_steps(
            env.depart[station, ordered]
        )

    dt = max(float(POWER_TO_ENERGY), 1e-9)
    capacity = torch.clamp(capacity, min=1e-6)
    remaining = torch.clamp(remaining, min=1.0)
    soc_kwh = torch.clamp(soc_pct, 0.0, 100.0) * capacity / 100.0
    target_kwh = torch.clamp(target_pct, 0.0, 100.0) * capacity / 100.0
    physical_lower = torch.maximum(-max_power, -soc_kwh / dt)
    physical_upper = torch.minimum(max_power, (capacity - soc_kwh) / dt)
    future_charge_kwh = (
        torch.clamp(remaining - 1.0, min=0.0) * max_power * dt
    )
    safe_lower = (
        target_kwh
        - soc_kwh
        + max(float(departure_slack_kwh), 0.0)
        - future_charge_kwh
    ) / dt
    safe_lower = torch.maximum(physical_lower, safe_lower)
    safe_lower = torch.minimum(safe_lower, physical_upper)

    raw_power = torch.clamp(
        actions * max_power, physical_lower, physical_upper
    )
    raw_power = torch.where(present, raw_power, torch.zeros_like(raw_power))

    # A station does not rescue a local actor miss on its own.  This lower
    # correction bound only prevents a downward central request from making the
    # actor's current departure-reachability position worse.
    correction_lower = torch.minimum(raw_power, safe_lower)
    target_upper = torch.minimum(
        physical_upper, torch.clamp((target_kwh - soc_kwh) / dt, min=0.0)
    )

    deficit_kwh = torch.clamp(target_kwh - soc_kwh, min=0.0)
    deliverable_kwh = torch.clamp(remaining * max_power * dt, min=1e-6)
    pressure = torch.clamp(deficit_kwh / deliverable_kwh, 0.0, 2.0)
    remaining_fraction = torch.clamp(
        remaining / max(float(getattr(env, "episode_steps", 288)), 1.0),
        0.0,
        1.0,
    )
    toward_target_priority = torch.where(
        present,
        0.10
        + 4.0 * pressure
        + deficit_kwh / capacity
        + 1.0 / remaining,
        torch.zeros_like(pressure),
    )
    energy_slack_fraction = torch.clamp(
        (soc_kwh - target_kwh) / capacity, 0.0, 1.0
    )
    empty_capacity_fraction = torch.clamp(
        (capacity - soc_kwh) / capacity, 0.0, 1.0
    )
    flexible_priority = torch.where(
        present,
        0.10
        + torch.clamp(1.0 - pressure, 0.0, 1.0)
        + remaining_fraction
        + 2.0 * energy_slack_fraction,
        torch.zeros_like(pressure),
    )
    surplus_charge_priority = torch.where(
        present,
        0.10 + remaining_fraction + empty_capacity_fraction,
        torch.zeros_like(pressure),
    )

    toward_target_headroom = torch.clamp(target_upper - raw_power, min=0.0)
    after_target = raw_power + toward_target_headroom
    surplus_headroom = torch.clamp(physical_upper - after_target, min=0.0)

    non_discharge_floor = torch.maximum(
        correction_lower, torch.zeros_like(correction_lower)
    )
    remove_charge_headroom = torch.clamp(
        raw_power - non_discharge_floor, min=0.0
    )
    after_remove_charge = raw_power - remove_charge_headroom
    safe_discharge_headroom = torch.clamp(
        after_remove_charge - correction_lower, min=0.0
    )

    raw_station = raw_power.sum(dim=1)
    target_up_station = toward_target_headroom.sum(dim=1)
    surplus_up_station = surplus_headroom.sum(dim=1)
    remove_charge_station = remove_charge_headroom.sum(dim=1)
    safe_discharge_station = safe_discharge_headroom.sum(dim=1)

    envelope = StationFlexibilityEnvelope(
        raw_power_kw=raw_station,
        safe_min_power_kw=(
            raw_station - remove_charge_station - safe_discharge_station
        ),
        safe_max_power_kw=(
            raw_station + target_up_station + surplus_up_station
        ),
        toward_target_charge_headroom_kw=target_up_station,
        surplus_charge_headroom_kw=surplus_up_station,
        remove_charge_headroom_kw=remove_charge_station,
        safe_discharge_headroom_kw=safe_discharge_station,
        toward_target_priority=_aggregate_priority(
            toward_target_headroom, toward_target_priority
        ),
        surplus_charge_priority=_aggregate_priority(
            surplus_headroom, surplus_charge_priority
        ),
        remove_charge_priority=_aggregate_priority(
            remove_charge_headroom, flexible_priority
        ),
        safe_discharge_priority=_aggregate_priority(
            safe_discharge_headroom, flexible_priority
        ),
    )
    envelope.validate()
    return _StationLocalDispatchContext(
        actor_actions=actions,
        present=present,
        max_power_kw=max_power,
        raw_power_kw=raw_power,
        physical_upper_kw=physical_upper,
        target_upper_kw=target_upper,
        correction_lower_kw=correction_lower,
        toward_target_priority=toward_target_priority,
        flexible_priority=flexible_priority,
        surplus_charge_priority=surplus_charge_priority,
        envelope=envelope,
    )


def allocate_central_station_targets(
    envelope: StationFlexibilityEnvelope,
    *,
    request_kw: float,
    tracking_enabled: bool,
    enabled: bool = True,
    waterfill_iterations: int = 16,
) -> tuple[torch.Tensor, dict]:
    """Choose station powers using aggregate station information only."""
    envelope.validate()
    station_targets = envelope.raw_power_kw.clone()
    requested_correction = 0.0
    tier_allocations = {
        "toward_target_charge_kw": 0.0,
        "surplus_charge_kw": 0.0,
        "remove_charge_kw": 0.0,
        "safe_discharge_kw": 0.0,
    }
    _tier_tensors: dict[str, torch.Tensor] = {}

    if bool(enabled) and bool(tracking_enabled):
        # One read, not two: nothing moves station_targets between them, so the
        # second call re-synchronised the device for a number already in hand.
        residual = float(request_kw) - float(station_targets.sum().item())
        requested_correction = residual
        if residual > 1e-7:
            add = capped_weighted_waterfill(
                envelope.toward_target_charge_headroom_kw,
                envelope.toward_target_priority,
                residual,
                iterations=waterfill_iterations,
            )
            station_targets = station_targets + add
            _tier_tensors["toward_target_charge_kw"] = add.sum()
            residual = float(request_kw) - float(station_targets.sum().item())
            if residual > 1e-7:
                add = capped_weighted_waterfill(
                    envelope.surplus_charge_headroom_kw,
                    envelope.surplus_charge_priority,
                    residual,
                    iterations=waterfill_iterations,
                )
                station_targets = station_targets + add
                _tier_tensors["surplus_charge_kw"] = add.sum()
        elif residual < -1e-7:
            subtract = capped_weighted_waterfill(
                envelope.remove_charge_headroom_kw,
                envelope.remove_charge_priority,
                -residual,
                iterations=waterfill_iterations,
            )
            station_targets = station_targets - subtract
            _tier_tensors["remove_charge_kw"] = subtract.sum()
            residual = float(request_kw) - float(station_targets.sum().item())
            if residual < -1e-7:
                subtract = capped_weighted_waterfill(
                    envelope.safe_discharge_headroom_kw,
                    envelope.safe_discharge_priority,
                    -residual,
                    iterations=waterfill_iterations,
                )
                station_targets = station_targets - subtract
                _tier_tensors["safe_discharge_kw"] = subtract.sum()

    station_targets = torch.maximum(
        envelope.safe_min_power_kw,
        torch.minimum(station_targets, envelope.safe_max_power_kw),
    )
    station_correction = station_targets - envelope.raw_power_kw
    # The tier figures are diagnostics and the two below are read once each, so
    # they all cross together rather than stalling the device one at a time.
    _keys = list(_tier_tensors)
    _stacked = torch.stack(
        [station_correction.sum(),
         (torch.abs(station_correction) > 1e-5).sum().to(station_correction.dtype)]
        + [_tier_tensors[k] for k in _keys]
    ).tolist()
    for _offset, _key in enumerate(_keys):
        tier_allocations[_key] = float(_stacked[2 + _offset])
    return station_targets, {
        "requested_correction_kw": float(requested_correction),
        "planned_correction_kw": float(_stacked[0]),
        "corrected_station_count": int(round(_stacked[1])),
        "tier_allocations_kw": tier_allocations,
        "waterfill_iterations": int(waterfill_iterations),
    }


def dispatch_station_targets_to_evs(
    context: _StationLocalDispatchContext,
    station_targets_kw: torch.Tensor,
    *,
    waterfill_iterations: int = 16,
) -> torch.Tensor:
    """Map each station target to its own EVs using private local state."""
    envelope = context.envelope
    if station_targets_kw.shape != envelope.raw_power_kw.shape:
        raise ValueError("one central target power is required per station")

    central_power = context.raw_power_kw.clone()
    station_delta = station_targets_kw - envelope.raw_power_kw

    positive_amount = torch.clamp(station_delta, min=0.0)
    toward_target_headroom = torch.clamp(
        context.target_upper_kw - central_power, min=0.0
    )
    toward_target_amount = torch.minimum(
        positive_amount, envelope.toward_target_charge_headroom_kw
    )
    add = _batched_capped_weighted_waterfill(
        toward_target_headroom,
        context.toward_target_priority,
        toward_target_amount,
        iterations=waterfill_iterations,
    )
    central_power = central_power + add
    positive_amount = torch.clamp(positive_amount - add.sum(dim=1), min=0.0)
    surplus_headroom = torch.clamp(
        context.physical_upper_kw - central_power, min=0.0
    )
    add = _batched_capped_weighted_waterfill(
        surplus_headroom,
        context.surplus_charge_priority,
        positive_amount,
        iterations=waterfill_iterations,
    )
    central_power = central_power + add

    negative_amount = torch.clamp(-station_delta, min=0.0)
    non_discharge_floor = torch.maximum(
        context.correction_lower_kw,
        torch.zeros_like(context.correction_lower_kw),
    )
    remove_charge_headroom = torch.clamp(
        central_power - non_discharge_floor, min=0.0
    )
    remove_charge_amount = torch.minimum(
        negative_amount, envelope.remove_charge_headroom_kw
    )
    subtract = _batched_capped_weighted_waterfill(
        remove_charge_headroom,
        context.flexible_priority,
        remove_charge_amount,
        iterations=waterfill_iterations,
    )
    central_power = central_power - subtract
    negative_amount = torch.clamp(
        negative_amount - subtract.sum(dim=1), min=0.0
    )
    safe_discharge_headroom = torch.clamp(
        central_power - context.correction_lower_kw, min=0.0
    )
    subtract = _batched_capped_weighted_waterfill(
        safe_discharge_headroom,
        context.flexible_priority,
        negative_amount,
        iterations=waterfill_iterations,
    )
    central_power = central_power - subtract
    return torch.where(
        context.present, central_power, torch.zeros_like(central_power)
    )


def allocate_central_ev_residual(
    env,
    actor_actions: torch.Tensor,
    *,
    request_kw: float,
    tracking_enabled: bool,
    enabled: bool = True,
    departure_slack_kwh: float = 0.1,
    waterfill_iterations: int = 16,
) -> tuple[torch.Tensor, dict]:
    """Compatibility wrapper executing the full hierarchical controller."""
    context = build_station_flexibility_envelope(
        env,
        actor_actions,
        departure_slack_kwh=departure_slack_kwh,
    )
    station_targets, central_info = allocate_central_station_targets(
        context.envelope,
        request_kw=request_kw,
        tracking_enabled=tracking_enabled,
        enabled=enabled,
        waterfill_iterations=waterfill_iterations,
    )
    central_power = dispatch_station_targets_to_evs(
        context,
        station_targets,
        waterfill_iterations=waterfill_iterations,
    )

    corrected_actions = torch.where(
        context.max_power_kw > 1e-8,
        central_power / torch.clamp(context.max_power_kw, min=1e-8),
        torch.zeros_like(central_power),
    )
    correction = central_power - context.raw_power_kw
    corrected_mask = context.present & (torch.abs(correction) > 1e-5)
    planned_station = central_power.sum(dim=1)
    station_target_error = planned_station - station_targets
    envelope = context.envelope
    # One transfer instead of six.  Each `.item()` is a separate device
    # synchronisation, and this dictionary is built on every step whether or
    # not anyone reads it -- the training loop passes build_info=False and
    # still pays for all of them.  Stacking the scalars keeps every value
    # bit-identical and costs one stall rather than six.
    _scalars = torch.stack([
        envelope.raw_power_kw.sum(),
        planned_station.sum(),
        correction.sum(),
        torch.abs(correction).sum(),
        corrected_mask.sum().to(correction.dtype),
        (torch.abs(station_target_error).max() if station_target_error.numel()
         else torch.zeros((), dtype=correction.dtype, device=correction.device)),
    ]).tolist()
    return torch.clamp(corrected_actions, -1.0, 1.0), {
        "raw_actor_ev_power_kw": context.raw_power_kw,
        "raw_actor_station_powers": envelope.raw_power_kw,
        "raw_actor_total_power_kw": float(_scalars[0]),
        "station_safe_min_power_kw": envelope.safe_min_power_kw,
        "station_safe_max_power_kw": envelope.safe_max_power_kw,
        "central_station_target_powers": station_targets,
        "planned_central_ev_power_kw": central_power,
        "planned_central_station_powers": planned_station,
        "planned_central_total_power_kw": float(_scalars[1]),
        "requested_correction_kw": central_info["requested_correction_kw"],
        "planned_correction_kw": float(_scalars[2]),
        "absolute_correction_kw": float(_scalars[3]),
        "corrected_ev_count": int(round(_scalars[4])),
        "corrected_station_count": central_info["corrected_station_count"],
        "station_target_max_abs_error_kw": float(_scalars[5]),
        "tier_allocations_kw": central_info["tier_allocations_kw"],
        # Kept for old logging consumers.  The hierarchical layer never forces
        # an actor upward to its reachability floor on its own.
        "safety_floor_correction_kw": 0.0,
        "waterfill_iterations": int(waterfill_iterations),
        "architecture": "station_envelope_hierarchical",
    }
