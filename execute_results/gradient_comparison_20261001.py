"""Aggregate recorded AB gradient diagnostics for AEMO and ERCOT."""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import json
import math
from pathlib import Path
import statistics
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]
TAGS = [
    "Gradient/actor_source_local_raw_mean",
    "Gradient/actor_source_global_raw_mean",
    "Gradient/actor_source_global_ratio_mean",
    "Gradient/actor_source_cos_mean",
    "Gradient/actor_source_cos_valid_fraction_mean",
    "Gradient/global_critic_raw",
    "Clipping/global_critic",
    "GlobalCritic/current_q_abs_mean",
    "GlobalCritic/target_q_abs_mean",
    "GlobalCritic/td_error_abs_mean",
    "GlobalCritic/reward_scale",
    "GlobalCritic/reward_baseline",
    "GlobalCritic/reward_raw_abs_mean",
    "GlobalCritic/reward_term_abs_mean",
    "GlobalCritic/td_target_abs_mean",
]

def read_events(run):
    data = {}
    for path in sorted((ROOT / "archive" / run / "performance").glob("events.out.tfevents.*")):
        accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags()["scalars"]:
            if not tag.startswith(("Gradient/", "Clipping/", "GlobalCritic/", "Q/")):
                continue
            values = data.setdefault(tag, {})
            for event in accumulator.Scalars(tag):
                previous = values.get(event.step)
                if previous is None or event.wall_time > previous[0]:
                    values[event.step] = (event.wall_time, event.value)
    return data

def stats(data, tag, lo, hi):
    values = [value for step, (_, value) in data.get(tag, {}).items() if lo <= step <= hi]
    finite = [value for value in values if math.isfinite(value)]
    return {
        "count": len(values),
        "nonfinite": len(values) - len(finite),
        "mean": statistics.mean(finite) if finite else None,
        "min": min(finite) if finite else None,
        "max": max(finite) if finite else None,
    }

def summarize(data, lo, hi):
    result = {"episodes": [lo, hi], "metrics": {tag: stats(data, tag, lo, hi) for tag in TAGS}}
    for kind in ("actor", "local_critic"):
        agents = [stats(data, f"Clipping/{kind}_agent{i}", lo, hi) for i in range(1, 8)]
        result[f"{kind}_clip_fraction"] = statistics.mean(row["mean"] for row in agents if row["mean"] is not None)
        result[f"{kind}_clip_fraction_by_agent"] = [row["mean"] for row in agents]
    result["global_ratio_by_agent"] = [stats(data, f"Gradient/actor_source_global_ratio_agent{i}", lo, hi)["mean"] for i in range(1, 8)]
    result["recorded_nonfinite"] = sum(stats(data, tag, lo, hi)["nonfinite"] for tag in data)
    return result

aemo = read_events("prod_aemoplan_AB_7station_20260926_210314")
ercot = read_events("prod_ercotplan_AB_7station_20260930_221551")
result = {
    "aemo_matched_episodes": summarize(aemo, 1175, 1274),
    "ercot_matched_episodes": summarize(ercot, 1175, 1274),
    "aemo_last_100": summarize(aemo, 2108, 2207),
}
output = ROOT / "execute_results/gradient_comparison_20261001.json"
output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
