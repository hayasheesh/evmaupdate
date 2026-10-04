"""完全情報LPで、360組それぞれに「帯とSoC目標を全部満たす充放電が存在するか」を判定する。

判定は入札段階の未見指令評価（tools/evaluate_unseen_bid_feasibility.py）と同じ
certify と fixed_bid_tracking_bands を使う。EV実現は、MARLの評価と同じ手順
（同じ環境を作り、同じ seed を入れて同じ引数で reset）で出力0の1日を流し、
到着したEVを取り出す。制御の評価の「目標±許容幅」と帯が一致するかも1組ずつ照合する。
"""
from __future__ import annotations

import csv
import json
import time

import numpy as np

import common

INFO = common.configure()

import torch  # noqa: E402

from Config import EPISODE_STEPS  # noqa: E402
from environment.EVEnv import EVEnv  # noqa: E402
from environment.normalize import use_instruction_scale  # noqa: E402
from market.physical_lp_bidding.colgen_feasibility import certify  # noqa: E402
from market.physical_lp_bidding.data_classes import BiddingLPConfig, EVSpec  # noqa: E402
from market.physical_lp_bidding.joint_validation import fixed_bid_tracking_bands  # noqa: E402
from tools.evaluator import set_env_seed  # noqa: E402
from training.lower_bid_training import build_fixed_upper_bid_training_episode  # noqa: E402

TIME_LIMIT_S = 20.0


def capture_evs(env: EVEnv) -> list[EVSpec]:
    """evenv_adapter.sample_ev_specs_from_evenv と同じ取り出し方（reset 済みの env から）。"""
    seen: set[tuple[int, int]] = set()
    specs: list[EVSpec] = []

    def capture() -> None:
        for station in range(env.num_stations):
            for slot_tensor in torch.nonzero(env.ev_mask[station], as_tuple=False).squeeze(-1):
                slot = int(slot_tensor.item())
                key = (int(env.ev_ids[station, slot].item()), int(env.arrival_step[station, slot].item()))
                if key in seen:
                    continue
                seen.add(key)
                arrival = max(key[1] - 1, 0)
                raw_departure = int(env.depart[station, slot].item())
                specs.append(EVSpec(
                    arrival_t=int(np.clip(arrival, 0, EPISODE_STEPS - 1)),
                    departure_t=max(raw_departure, arrival + 1),
                    initial_soc=float(env.soc[station, slot].item()),
                    target_soc=float(env.target[station, slot].item()),
                    capacity_kwh=float(env.ev_capacity_kwh[station, slot].item()),
                    max_charge_kw=float(env.ev_max_power_kw[station, slot].item()),
                    max_discharge_kw=float(env.ev_max_power_kw[station, slot].item()),
                    station_id=int(station),
                    ev_id=f'{key[0]}@{key[1]}',
                    target_required=bool(raw_departure <= EPISODE_STEPS),
                ))

    capture()
    actions = torch.zeros((env.num_stations, env.max_ev_per_station), dtype=torch.float32, device=env.soc.device)
    with torch.no_grad():
        for _ in range(EPISODE_STEPS):
            env.begin_step()
            capture()
            _o, _l, _g, done, _i = env.apply_action(actions, build_info=False, return_observation=False)
            if bool(np.all(np.asarray(done, dtype=bool))):
                break
    return specs


def main() -> None:
    rows = []
    started = time.perf_counter()
    for day_index, entry, fixed_bid in common.day_cases():
        cfg = BiddingLPConfig(
            assessment_band_fraction=float(fixed_bid['assessment_band_fraction']),
            apply_transition_band=bool(fixed_bid['apply_transition_band']),
        )
        baseline = np.asarray(fixed_bid['baseline_plan'], dtype=float)
        up = np.asarray(fixed_bid['up_plan'], dtype=float)
        down = np.asarray(fixed_bid['down_plan'], dtype=float)
        env = EVEnv()
        for s, command in enumerate(fixed_bid['activation_scenario_payload']):
            target, tol, arr, _info = build_fixed_upper_bid_training_episode(fixed_bid, s)
            target = np.asarray(target, dtype=float).reshape(-1)[:EPISODE_STEPS]
            tol = np.asarray(tol, dtype=float).reshape(-1)[:EPISODE_STEPS]
            enabled = np.asarray(arr.get('tracking_enabled_series', np.ones(EPISODE_STEPS, dtype=bool)),
                                 dtype=bool).reshape(-1)[:EPISODE_STEPS]
            _t, _tol, lower, upper = fixed_bid_tracking_bands(
                cfg, baseline, up, down,
                np.asarray(command['up_proxy'], dtype=float),
                np.asarray(command['down_proxy'], dtype=float),
                apply_transition_band=cfg.apply_transition_band,
            )
            bid_assessed = np.isfinite(lower) & np.isfinite(upper)
            band_lower_diff = float(np.max(np.abs(lower[enabled] - (target - tol)[enabled]))) if enabled.any() else 0.0
            band_upper_diff = float(np.max(np.abs(upper[enabled] - (target + tol)[enabled]))) if enabled.any() else 0.0
            for k in range(common.EV_SEEDS):
                seed = common.realized_seed(day_index, s, k)
                set_env_seed(seed)
                use_instruction_scale(arr.get('instruction_scale_kw', 1.0))
                env.reset(
                    net_demand_series=target,
                    tol_narrow_series=tol,
                    tracking_enabled_series=enabled,
                    market_context_series=arr.get('market_context_series'),
                    arrival_probabilities_by_station=fixed_bid.get('arrival_probabilities_by_station'),
                    day_context=fixed_bid.get('day_context'),
                )
                evs = capture_evs(env)
                t0 = time.perf_counter()
                feasible, _, info = certify(
                    evs, lower, upper, steps=cfg.steps, dt=cfg.dt_hours, eta_ch=cfg.eta_ch,
                    return_dispatch=False, time_limit_s=TIME_LIMIT_S,
                )
                rows.append({
                    'bid_day_index': day_index, 'service_date': entry['service_date'],
                    'scenario': s, 'seed': k, 'realized_seed': seed,
                    'command_source': str(command.get('source') or f"{command.get('source_date', '')}|{command.get('source_bmu', '')}"),
                    'evs': len(evs), 'evs_target_required': sum(ev.target_required for ev in evs),
                    'assessed_steps_env': int(enabled.sum()), 'assessed_steps_bid': int(bid_assessed.sum()),
                    'assessed_mask_equal': bool(np.array_equal(enabled, bid_assessed)),
                    'band_lower_max_diff_kw': band_lower_diff, 'band_upper_max_diff_kw': band_upper_diff,
                    'feasible': feasible, 'reason': info.get('reason') if feasible is None else '',
                    'certify_s': time.perf_counter() - t0,
                })
            done = sum(1 for r in rows if r['bid_day_index'] == day_index)
            print(f"day {day_index} {entry['service_date']} scenario {s + 1}/24 rows {done} "
                  f"feasible {sum(r['feasible'] is True for r in rows)}/{len(rows)} "
                  f"elapsed {time.perf_counter() - started:.0f}s", flush=True)
    with open(common.HERE / 'lp_certify_cases.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        'cases': len(rows),
        'feasible': sum(r['feasible'] is True for r in rows),
        'infeasible': sum(r['feasible'] is False for r in rows),
        'unknown': sum(r['feasible'] is None for r in rows),
        'assessed_mask_equal_all': all(r['assessed_mask_equal'] for r in rows),
        'band_max_diff_kw': max(max(r['band_lower_max_diff_kw'], r['band_upper_max_diff_kw']) for r in rows),
        'solver_time_limit_s': TIME_LIMIT_S,
        'pinned_source_sha256': INFO['pinned_source_sha256'],
        'elapsed_s': time.perf_counter() - started,
    }
    (common.HERE / 'lp_certify_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
