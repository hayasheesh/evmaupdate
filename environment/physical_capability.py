"""Physical aggregate power available from the currently connected EVs."""

from __future__ import annotations

import torch


SUSTAIN_HOURS = 0.5


def aggregate_physical_capability(env, block_end_step: int) -> tuple[float, float]:
    """Return the EV fleet's sustained ``(charge_kw, discharge_kw)`` for one block.

    Assessment I, judged on the realized state with the definition the bid uses
    (market.sustained_capability): only EVs connected until ``block_end_step``
    (the step count of the block's last step) count, each EV can discharge only
    the energy above what it still needs at the block end to reach its target
    by charging at its rated power afterwards, and each holds its rate for
    SUSTAIN_HOURS. The fleet holds any power between -discharge and +charge;
    the bid meets Assessment I when baseline - up and baseline + down stay
    inside it. The residual battery does not count, as in the bid.
    """

    from EnvConfig import POWER_TO_ENERGY

    mask = env.ev_mask & (env.depart >= int(block_end_step))
    if not bool(mask.any().item()):
        return 0.0, 0.0
    energy_kwh = env.soc / 100.0 * env.ev_capacity_kwh
    target_kwh = env.target / 100.0 * env.ev_capacity_kwh
    after_hours = torch.clamp(env.depart.float() - float(block_end_step), min=0.0) * float(POWER_TO_ENERGY)
    floor_kwh = torch.clamp(target_kwh - env.ev_max_power_kw * after_hours, min=0.0)
    horizon = max(float(SUSTAIN_HOURS), 1e-9)
    charge = torch.minimum(
        env.ev_max_power_kw, torch.clamp(env.ev_capacity_kwh - energy_kwh, min=0.0) / horizon
    )
    discharge = torch.minimum(
        env.ev_max_power_kw, torch.clamp(energy_kwh - floor_kwh, min=0.0) / horizon
    )
    charge = torch.where(mask, charge, torch.zeros_like(charge))
    discharge = torch.where(mask, discharge, torch.zeros_like(discharge))
    return float(charge.sum().item()), float(discharge.sum().item())


def step_dispatch_envelope(env, *, force_slack_kwh: float = 0.0) -> tuple[float, float, float]:
    """Return the ``(min_kw, max_kw, obligated_min_kw)`` reachable in one step.

    ``aggregate_physical_capability`` answers the Assessment-I question -- can
    the fleet hold a width for the whole product block -- so it spreads the
    stored energy over the thirty-minute horizon.  A single five-minute
    instruction is a different question, and the two disagree in both
    directions: one step may draw more power than a sustained half hour allows,
    while the departure guarantee forbids discharges that the sustained figure
    still counts as available.

    The bounds mirror the clamps ``apply_action`` applies -- power is the action
    times the rated power, and the resulting SoC is clipped to [0, 100].
    ``obligated_min_kw`` additionally holds every EV at the floor
    ``apply_force_charging`` would impose, so the gap between the two minima is
    what the departure guarantee costs at this instant.

    Signs follow the PCC convention used everywhere else: positive is import.
    """

    from EnvConfig import POWER_TO_ENERGY

    dt = max(float(POWER_TO_ENERGY), 1e-9)
    bess_min, bess_max = (
        env.bess_feasible_power_bounds_kw()
        if bool(getattr(env, "use_residual_bess", False))
        else (0.0, 0.0)
    )

    mask = env.ev_mask
    if not bool(mask.any().item()):
        return float(bess_min), float(bess_max), float(bess_min)

    zeros = torch.zeros_like(env.soc)
    soc = env.soc
    kwh_per_pct = env.ev_kwh_per_soc_pct
    rated = env.ev_max_power_kw

    charge_cap = torch.minimum(rated, (100.0 - soc) * kwh_per_pct / dt)
    discharge_cap = torch.minimum(rated, soc * kwh_per_pct / dt)
    charge_cap = torch.where(mask, charge_cap, zeros).clamp(min=0.0)
    discharge_cap = torch.where(mask, discharge_cap, zeros).clamp(min=0.0)

    # The same four skip conditions apply_force_charging uses, so an EV it
    # would leave alone keeps its full discharge available here too.
    max_step_kwh = torch.clamp(rated * dt, min=1e-9)
    need_kwh = (env.target - soc).clamp(min=0.0) * kwh_per_pct + float(
        max(0.0, force_slack_kwh)
    )
    remaining = env.depart.to(torch.float32) - float(env.step_count) + 1.0
    floor_kwh = need_kwh - (remaining - 1.0) * max_step_kwh
    forced = mask & (need_kwh > 0.0) & (remaining > 0.0) & (floor_kwh > -max_step_kwh)
    forced_floor_kw = torch.minimum(floor_kwh, max_step_kwh) / dt
    obligated_kw = torch.where(forced, forced_floor_kw, -discharge_cap)
    obligated_kw = torch.minimum(torch.maximum(obligated_kw, -discharge_cap), charge_cap)
    obligated_kw = torch.where(mask, obligated_kw, zeros)

    station_max = charge_cap.sum(dim=1)
    station_min = -discharge_cap.sum(dim=1)
    station_obligated = obligated_kw.sum(dim=1)

    return (
        float(station_min.sum().item()) + float(bess_min),
        float(station_max.sum().item()) + float(bess_max),
        float(station_obligated.sum().item()) + float(bess_min),
    )
