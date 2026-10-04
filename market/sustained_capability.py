"""Per-block sustained delivery capability of an EV fleet.

Secondary Reserve 2 requires a resource to sustain its awarded delta-kW for the
full 30-minute product block (EPRX product table: 応動時間 5 分以内 / 継続時間
30 分). Without this constraint the bidder submits widths the fleet cannot
physically hold. This module computes what the fleet can actually sustain for
Assessment I.

Energy lives in individual batteries and cannot move between vehicles, so the
aggregate sustained capability is exactly the sum of the per-vehicle sustained
capabilities -- the Minkowski sum of the per-EV sets collapses to a plain sum.
This is an equality, not a bound.

Two envelopes bracket where a vehicle's energy can be at the start of a block,
following the virtual-battery construction used for EV aggregation (Hagstrom &
Herre, arXiv:2511.19715):

``e_hi``  charge flat out from arrival, capped by the battery -- the most energy
          the vehicle could be holding, so the most it can give back;
``e_lo``  charge as late as possible while still meeting the departure target --
          the least energy it could be holding, so the most room it has left.

Up and down capability are therefore each computed against their own favourable
envelope. They are separate per-direction upper bounds and are **not**
simultaneously attainable: a block cannot hold its energy both high and low at
once. A bid taking both directions in one block needs a coupling constraint,
as in the Nordic LER rule (Lunde et al., arXiv:2404.12818), which reserves a
fraction of one direction to keep the other deliverable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from market.bid_env import N_BLOCKS

STEPS_PER_BLOCK = 6
STEP_HOURS = 5.0 / 60.0
BLOCK_HOURS = STEPS_PER_BLOCK * STEP_HOURS


@dataclass(frozen=True)
class SustainedCapability:
    """Per-block sustained delivery capability, in kW.

    Every array is indexed by 30-minute block. ``up``/``down`` are the binding
    limits: the smaller of what the chargers can move and what the batteries
    can supply or absorb for a whole block.
    """

    up: np.ndarray
    down: np.ndarray
    up_power_limit: np.ndarray
    down_power_limit: np.ndarray
    up_energy_limit: np.ndarray
    down_energy_limit: np.ndarray
    occupancy: np.ndarray

    def as_summary(self) -> dict:
        def stat(values, mask):
            picked = np.asarray(values, dtype=float)[mask]
            if not picked.size:
                return {"min": 0.0, "median": 0.0, "max": 0.0}
            return {
                "min": float(np.min(picked)),
                "median": float(np.median(picked)),
                "max": float(np.max(picked)),
            }

        occupied = self.occupancy > 0
        return {
            "blocks_with_evs": int(np.count_nonzero(occupied)),
            "occupancy": stat(self.occupancy, occupied),
            "up_kw": stat(self.up, occupied),
            "down_kw": stat(self.down, occupied),
            "up_binding_energy_blocks": int(
                np.count_nonzero(
                    occupied & (self.up_energy_limit < self.up_power_limit - 1e-9)
                )
            ),
            "down_binding_energy_blocks": int(
                np.count_nonzero(
                    occupied & (self.down_energy_limit < self.down_power_limit - 1e-9)
                )
            ),
        }


def _energy_envelope(ev, step: int) -> tuple[float, float]:
    """Return (e_lo, e_hi) kWh the vehicle could hold entering ``step``."""

    capacity = max(float(ev.capacity_kwh), 0.0)
    initial = float(np.clip(ev.initial_kwh, 0.0, capacity))
    arrival = int(ev.arrival_t)
    departure = int(ev.departure_t)
    # A session extending beyond the service-day horizon still has a real
    # departure target.  Its backward-reachable reserve must therefore shape
    # the envelope even though the full departure lies outside the bidding LP.
    target = (
        float(np.clip(ev.target_kwh, 0.0, capacity))
        if ev.target_required or departure > int(step)
        else 0.0
    )
    charge_kw = max(float(ev.max_charge_kw), 0.0)
    discharge_kw = max(float(ev.max_discharge_kw), 0.0)
    elapsed = max(step - arrival, 0) * STEP_HOURS
    remaining = max(departure - step, 0) * STEP_HOURS

    # Charging flat out from arrival, capped by the battery.
    e_hi = min(capacity, initial + charge_kw * elapsed)
    # The lowest reachable energy: a vehicle cannot be emptied faster than its
    # discharge rate allows, and cannot drop below what the departure target
    # will still need. Without the discharge floor the envelope would credit a
    # full battery with a full battery's worth of room to absorb, by assuming
    # it had already been emptied for free.
    e_lo = max(
        0.0,
        initial - discharge_kw * elapsed,
        target - charge_kw * remaining,
    )
    e_lo = min(e_lo, e_hi)
    return e_lo, e_hi


def sustained_capability_for_scenario(
    evs,
    *,
    blocks: int = N_BLOCKS,
    duration_hours: float = BLOCK_HOURS,
) -> SustainedCapability:
    """Compute what a fleet can sustain for ``duration_hours`` in each block.

    ``evs`` is one scenario's list of EV sessions. A vehicle contributes to a
    block only if it is connected for the whole block: a bid must be
    deliverable for the entire product duration, and a vehicle that leaves
    midway cannot be relied on for it.
    """

    n_blocks = int(blocks)
    up = np.zeros(n_blocks)
    down = np.zeros(n_blocks)
    up_p = np.zeros(n_blocks)
    down_p = np.zeros(n_blocks)
    up_e = np.zeros(n_blocks)
    down_e = np.zeros(n_blocks)
    occupancy = np.zeros(n_blocks)
    hours = max(float(duration_hours), 1e-9)

    for block in range(n_blocks):
        start = block * STEPS_PER_BLOCK
        end = start + STEPS_PER_BLOCK
        for ev in evs:
            # Connected for the whole block, not merely at its start.
            if not (int(ev.arrival_t) <= start and int(ev.departure_t) >= end):
                continue
            occupancy[block] += 1.0
            charge_kw = max(float(ev.max_charge_kw), 0.0)
            discharge_kw = max(float(ev.max_discharge_kw), 0.0)
            capacity = max(float(ev.capacity_kwh), 0.0)
            e_lo_start, e_hi_start = _energy_envelope(ev, start)
            e_lo_end, _e_hi_end = _energy_envelope(ev, end)

            # Up: give back energy held above what the departure target still
            # needs at the end of the block.
            ev_up_energy = max(e_hi_start - e_lo_end, 0.0) / hours
            # Down: absorb energy into the room left in the battery.
            ev_down_energy = max(capacity - e_lo_start, 0.0) / hours

            # Each vehicle sustains the smaller of its own two limits, and the
            # fleet total is the sum of those. Taking min(sum power, sum
            # energy) instead would overstate the fleet, letting one vehicle's
            # spare energy excuse another's missing charger power.
            up[block] += min(discharge_kw, ev_up_energy)
            down[block] += min(charge_kw, ev_down_energy)

            up_p[block] += discharge_kw
            down_p[block] += charge_kw
            up_e[block] += ev_up_energy
            down_e[block] += ev_down_energy

    return SustainedCapability(
        up=up,
        down=down,
        up_power_limit=up_p,
        down_power_limit=down_p,
        up_energy_limit=up_e,
        down_energy_limit=down_e,
        occupancy=occupancy,
    )


def sustained_capability_quantile(
    scenarios,
    *,
    quantile: float,
    blocks: int = N_BLOCKS,
    duration_hours: float = BLOCK_HOURS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the per-block ``quantile`` of sustained up/down capability.

    A marginal quantile per block and direction is a first-order reading of the
    fleet, not a reliability guarantee: the requirement is that every block and
    direction hold *jointly*, and taking each margin separately does not
    control that joint probability. Lunde et al. measure exactly this failure
    -- their percentile-based "naive" bidder overshoots its 10% allowance at
    13.9% and would be excluded from the market. Use this to size and inspect
    the fleet; use a joint chance constraint to set a bid.
    """

    if not scenarios:
        raise ValueError("sustained capability needs at least one EV scenario")
    ups = []
    downs = []
    for evs in scenarios:
        capability = sustained_capability_for_scenario(
            evs, blocks=blocks, duration_hours=duration_hours
        )
        ups.append(capability.up)
        downs.append(capability.down)
    q = float(np.clip(quantile, 0.0, 1.0)) * 100.0
    return (
        np.percentile(np.vstack(ups), q, axis=0),
        np.percentile(np.vstack(downs), q, axis=0),
    )
