"""Adapters from EVEnv forecast samples to upper-bid scenarios."""

from __future__ import annotations

import numpy as np
import torch

from .data_classes import ActivationScenario, EVSpec


def sample_ev_specs_from_evenv(
    *,
    seed: int,
    arrival_probabilities_by_station=None,
    day_context=None,
    arrival_counts_by_station_step=None,
    initial_evs_by_station=None,
    service_date=None,
) -> list[EVSpec]:
    """Roll EVEnv with zero power and return every accepted EV session."""

    from Config import EPISODE_STEPS
    from environment.EVEnv import EVEnv
    from tools.evaluator import set_env_seed

    set_env_seed(int(seed))
    env = EVEnv()
    env.reset(
        net_demand_series=np.zeros(EPISODE_STEPS, dtype=np.float32),
        arrival_probabilities_by_station=arrival_probabilities_by_station,
        arrival_counts_by_station_step=arrival_counts_by_station_step,
        initial_evs_by_station=initial_evs_by_station,
        day_context=day_context,
        service_date=service_date,
    )

    # EVEnv reuses integer IDs after departure, so the arrival epoch is part of
    # the session identity.
    seen: set[tuple[int, int]] = set()
    specs: list[EVSpec] = []

    def capture_new_sessions() -> None:
        for station in range(env.num_stations):
            active_slots = torch.nonzero(
                env.ev_mask[station], as_tuple=False
            ).squeeze(-1)
            for slot_tensor in active_slots:
                slot = int(slot_tensor.item())
                ev_id = int(env.ev_ids[station, slot].item())
                env_arrival_step = int(
                    env.arrival_step[station, slot].item()
                )
                session_key = (ev_id, env_arrival_step)
                if session_key in seen:
                    continue
                seen.add(session_key)
                arrival = max(env_arrival_step - 1, 0)
                raw_departure = int(env.depart[station, slot].item())
                departure = max(raw_departure, arrival + 1)
                specs.append(
                    EVSpec(
                        arrival_t=int(
                            np.clip(arrival, 0, EPISODE_STEPS - 1)
                        ),
                        departure_t=departure,
                        # EVEnv SoC is in percent; EVSpec takes fractions.
                        initial_soc=float(env.soc[station, slot].item()) / 100.0,
                        target_soc=float(env.target[station, slot].item()) / 100.0,
                        capacity_kwh=float(
                            env.ev_capacity_kwh[station, slot].item()
                        ),
                        max_charge_kw=float(
                            env.ev_max_power_kw[station, slot].item()
                        ),
                        max_discharge_kw=float(
                            env.ev_max_power_kw[station, slot].item()
                        ),
                        station_id=int(station),
                        ev_id=f"{ev_id}@{env_arrival_step}",
                        target_required=bool(
                            raw_departure <= EPISODE_STEPS
                        ),
                    )
                )

    capture_new_sessions()
    actions = torch.zeros(
        (env.num_stations, env.max_ev_per_station),
        dtype=torch.float32,
        device=env.soc.device,
    )
    with torch.inference_mode():
        for _step in range(EPISODE_STEPS):
            env.begin_step()
            capture_new_sessions()
            _obs, _local, _global, done, _info = env.apply_action(
                actions,
                build_info=False,
                return_observation=False,
            )
            if bool(np.all(np.asarray(done, dtype=bool))):
                break
    return specs


def stratified_ev_activation_scenarios(
    *,
    ev_specs_by_scenario: list[list[EVSpec]],
    activation_payloads: list[dict],
    activations_per_ev: int,
) -> list[ActivationScenario]:
    """Build a balanced reduced cross of independent EV and command draws."""

    if not ev_specs_by_scenario:
        raise ValueError("at least one EV scenario is required")
    if not activation_payloads:
        raise ValueError("at least one activation scenario is required")
    activation_count = len(activation_payloads)
    count = max(1, min(int(activations_per_ev), activation_count))
    offsets = [
        int(np.floor(index * activation_count / count))
        for index in range(count)
    ]
    scenarios: list[ActivationScenario] = []
    for ev_index, evs in enumerate(ev_specs_by_scenario):
        for cross_slot, offset in enumerate(offsets):
            activation_index = (ev_index + offset) % activation_count
            payload = activation_payloads[activation_index]
            name = payload.get("name") or (
                f"activation_{activation_index:02d}"
            )
            scenarios.append(
                ActivationScenario(
                    name=(
                        f"cross_ev{ev_index:02d}_a{activation_index:02d}_"
                        f"{name}"
                    ),
                    up_signal=np.asarray(
                        payload.get("up_proxy"), dtype=float
                    ).reshape(-1).copy(),
                    down_signal=np.asarray(
                        payload.get("down_proxy"), dtype=float
                    ).reshape(-1).copy(),
                    # EVSpec is immutable. Every command for one day uses the
                    # same fixed EV realization, so sharing this list avoids
                    # duplicating thousands of objects per command.
                    evs=evs,
                    weight=float(payload.get("weight", 1.0)),
                    metadata={
                        "ev_scenario": ev_index,
                        "activation_scenario": activation_index,
                        "cross_slot": cross_slot,
                        "activation_source": payload.get("source", ""),
                    },
                )
            )
    return scenarios
