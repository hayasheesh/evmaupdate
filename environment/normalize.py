"""
Observation normalization helpers.

This module converts raw EVEnv observations into the scale expected by the
actor/critic networks, and provides small inverse helpers for diagnostics.

Input:
- `normalize_observation(obs)` receives a station-major observation with shape
  roughly `(num_stations, state_dim)`. `obs` may be a NumPy array from EVEnv or
  a torch Tensor from replay/evaluation utilities.
- Per-station layout is:
  `[EV slots][demand lookahead][market tracking enabled][current step]`

Output:
- `normalize_observation()` returns a torch Tensor with the same shape as input.
- The function clones the input tensor before editing, so callers keep their raw
  observation unchanged.

Normalization rules:
- `presence`: unchanged. Usually 0/1.
- `soc`: raw percent SoC `[0, 100]` -> `[0, 1]`.
- `remaining_time`: raw remaining controllable action steps `[0, EPISODE_STEPS]` -> `[0, 1]`.
- `needed_soc`: raw percent `[0, 100]` -> `[0, 1]`, then clamped to `[-1, 1]`.
- `battery_capacity_kwh` and `max_power_kw`: present only when
  `USE_HETEROGENEOUS_EV_PHYSICS=True`; raw values are scaled to `[0, 1]`.
- `switch_count`: raw count -> `[0, 1]` by dividing by `MAX_SWITCH_COUNT`.
- `last_direction`: clamped to `[-1, 1]`; expected values are charge/discharge
  direction-like indicators.
- `demand lookahead`: centered by the Config demand-target range and divided
  by half of that range, then clamped to `[-OBS_DEMAND_CLAMP,
  OBS_DEMAND_CLAMP]`.
- `market tracking enabled`: unchanged 0/1 participation indicator.
- `current step`: raw step `[0, EPISODE_STEPS]` -> `[0, 1]`.

Dependencies / layout contract:
- `environment.observation_config` defines EV feature order and tail layout.
  If `EV_FEATURE_NAMES`, `LOCAL_DEMAND_STEPS`, or `LOCAL_USE_STEP` changes,
  this file must remain consistent with that layout.
- `Config` supplies episode length, max EV slots per station, switch-count
  scale, and demand-adjustment min/max used for AG request scaling.

Notes:
- `denormalize_observation()` is a lightweight debug/plotting helper for the
  first station only. Training should use normalized tensors directly.
- Unknown EV features are passed through unchanged so adding already-normalized
  features does not require extra code here.
"""

import json
from pathlib import Path

import numpy as np
import torch

from EnvConfig import (
    OBS_DEMAND_CLAMP,
    EPISODE_STEPS,
    DEMAND_TARGET_MIN_KW,
    DEMAND_TARGET_MAX_KW,
    EV_CAPACITY_OBS_SCALE_KWH,
    EV_CHARGER_POWER_OBS_SCALE_KW,
    MAX_EV_PER_STATION,
    MAX_SWITCH_COUNT,
    PHYSICAL_MAX_POWER_KW,
    BESS_POWER_KW,
    BESS_CONTEXT_USE_OBS,
    LOCAL_USE_FLEET_RESIDUAL,
)
from environment.observation_config import (
    EV_FEAT_DIM,
    EV_FEATURE_NAMES,
    LOCAL_DEMAND_STEPS,
    LOCAL_USE_TRACKING_ENABLED,
    LOCAL_USE_STEP,
    BID_CONTEXT_FEATURES,
    BESS_CONTEXT_FEATURES,
    LOWER_BID_CONTEXT_USE_OBS,
)


OBSERVATION_NORMALIZATION_FILENAME = "observation_normalization.json"
OBSERVATION_NORMALIZATION_VERSION = 1

_DEFAULT_OBSERVATION_NORMALIZATION = {
    "version": OBSERVATION_NORMALIZATION_VERSION,
    "source": "physical_defaults",
    "quantile": None,
    "demand_center_kw": float(
        (DEMAND_TARGET_MIN_KW + DEMAND_TARGET_MAX_KW) / 2.0
    ),
    "demand_scale_kw": float(
        (DEMAND_TARGET_MAX_KW - DEMAND_TARGET_MIN_KW) / 2.0
    ),
    "instruction_scale_scale_kw": float(PHYSICAL_MAX_POWER_KW),
}
_observation_normalization = dict(_DEFAULT_OBSERVATION_NORMALIZATION)
AG_REQUEST_CENTER = float(_observation_normalization["demand_center_kw"])
AG_REQUEST_SCALE = float(_observation_normalization["demand_scale_kw"])


def _valid_scale(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive value, got {value!r}")
    return value


def configure_observation_normalization(profile=None):
    """Install one immutable-by-convention normalization profile in-process."""

    global _observation_normalization, AG_REQUEST_CENTER, AG_REQUEST_SCALE
    merged = dict(_DEFAULT_OBSERVATION_NORMALIZATION)
    if profile is not None:
        merged.update(dict(profile))
    if int(merged.get("version", -1)) != OBSERVATION_NORMALIZATION_VERSION:
        raise ValueError(
            "unsupported observation normalization version: "
            f"{merged.get('version')!r}"
        )
    center = float(merged["demand_center_kw"])
    if not np.isfinite(center):
        raise ValueError("demand_center_kw must be finite")
    scale_keys = (
        "demand_scale_kw",
        "instruction_scale_scale_kw",
    )
    for key in scale_keys:
        merged[key] = _valid_scale(merged[key], key)
    merged["demand_center_kw"] = center
    _observation_normalization = merged
    AG_REQUEST_CENTER = center
    AG_REQUEST_SCALE = float(merged["demand_scale_kw"])
    return dict(_observation_normalization)


def get_observation_normalization_profile():
    return dict(_observation_normalization)


def _finite_values(values):
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return array[np.isfinite(array)]


def _quantile_scale(values, quantile):
    array = _finite_values(values)
    if array.size == 0:
        return 1.0
    return max(float(np.quantile(np.abs(array), float(quantile))), 1.0)


def derive_observation_normalization_profile(samples, *, quantile=0.95, source="train_bid_bank"):
    """Derive zero-centred physical scales from training-bank samples only."""

    quantile = float(quantile)
    if not 0.5 <= quantile <= 1.0:
        raise ValueError("normalization quantile must be in [0.5, 1.0]")
    feature_to_profile_key = {
        "demand_kw": "demand_scale_kw",
        "instruction_scale_kw": "instruction_scale_scale_kw",
    }
    profile = {
        "version": OBSERVATION_NORMALIZATION_VERSION,
        "source": str(source),
        "quantile": quantile,
        "demand_center_kw": 0.0,
        "sample_counts": {},
    }
    for feature_name, profile_key in feature_to_profile_key.items():
        values = _finite_values(samples.get(feature_name, []))
        profile[profile_key] = _quantile_scale(values, quantile)
        profile["sample_counts"][feature_name] = int(values.size)
    return configure_observation_normalization(profile)


def use_instruction_scale(scale_kw):
    """Point the demand-series scale at one day's own instruction envelope.

    The observation carries the instruction as ``demand_0``. Its magnitude is
    set by that day's submitted bid, which differs day to day, so a single
    bank-wide scale leaves narrow days squashed near zero and wide days near
    the clamp. Call this with the day's envelope before resetting the
    environment; the rest of the profile is untouched.
    """

    profile = dict(get_observation_normalization_profile())
    profile["demand_scale_kw"] = max(float(scale_kw), 1.0)
    return configure_observation_normalization(profile)


def save_observation_normalization_profile(path, profile=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = configure_observation_normalization(
        get_observation_normalization_profile() if profile is None else profile
    )
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temp_path.replace(path)
    return path


def load_observation_normalization_profile(path):
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return configure_observation_normalization(payload)


def load_observation_normalization_for_archive(path):
    """Load a saved run profile; restore physical defaults when none exists."""

    path = Path(path).resolve()
    base = path.parent if path.is_file() else path
    candidates = []
    for root in (base, *base.parents):
        candidates.extend((
            root / "input" / OBSERVATION_NORMALIZATION_FILENAME,
            root / OBSERVATION_NORMALIZATION_FILENAME,
        ))
        if len(candidates) >= 8:
            break
    for candidate in candidates:
        if candidate.is_file():
            return load_observation_normalization_profile(candidate)
    configure_observation_normalization(None)
    return None


def _market_context_scale(feature_name):
    profile_key = {
        "instruction_scale_kw": "instruction_scale_scale_kw",
    }.get(feature_name)
    if profile_key is None:
        return 1.0
    return float(_observation_normalization[profile_key])


def _normalize_bess_context(feature_name, value):
    if feature_name == "bess_soc_pct":
        return torch.clamp(value / 100.0, 0.0, 1.0)
    if feature_name == "last_bess_power_kw":
        scale = max(float(BESS_POWER_KW), 1.0)
    elif feature_name == "last_pcc_power_kw":
        scale = max(float(PHYSICAL_MAX_POWER_KW) + float(BESS_POWER_KW), 1.0)
    else:
        scale = max(float(PHYSICAL_MAX_POWER_KW), 1.0)
    return torch.clamp(
        value / scale,
        -float(OBS_DEMAND_CLAMP),
        float(OBS_DEMAND_CLAMP),
    )


def _denormalize_bess_context(feature_name, value):
    if feature_name == "bess_soc_pct":
        return value * 100.0
    if feature_name == "last_bess_power_kw":
        return value * max(float(BESS_POWER_KW), 1.0)
    if feature_name == "last_pcc_power_kw":
        return value * max(float(PHYSICAL_MAX_POWER_KW) + float(BESS_POWER_KW), 1.0)
    return value * max(float(PHYSICAL_MAX_POWER_KW), 1.0)


def _to_tensor(obs):
    # EVEnv usually emits NumPy arrays, while replay/test paths may already use
    # tensors. The rest of this module uses torch indexing and torch.clamp.
    if isinstance(obs, np.ndarray):
        return torch.from_numpy(obs).float()
    return obs


def _feature_index(name):
    # Optional lookup: diagnostic code can keep working even if a feature is
    # removed from EV_FEATURE_NAMES.
    try:
        return EV_FEATURE_NAMES.index(name)
    except ValueError:
        return None


def _normalize_ev_feature(name, value):
    # Each EV slot column has a different physical unit, so normalization is
    # feature-name based rather than position-only.
    if name == "presence":
        return value
    if name == "soc":
        return value / 100.0
    if name == "remaining_time":
        return value / EPISODE_STEPS
    if name == "needed_soc":
        return torch.clamp(value / 100.0, -1.0, 1.0)
    if name == "battery_capacity_kwh":
        return torch.clamp(value / max(float(EV_CAPACITY_OBS_SCALE_KWH), 1.0), 0.0, 1.0)
    if name == "max_power_kw":
        return torch.clamp(value / max(float(EV_CHARGER_POWER_OBS_SCALE_KW), 1.0), 0.0, 1.0)
    if name == "switch_count":
        return torch.clamp(value / max(float(MAX_SWITCH_COUNT), 1.0), 0.0, 1.0)
    if name == "last_direction":
        return torch.clamp(value, -1.0, 1.0)
    return value


def _denormalize_ev_feature(name, value):
    # Inverse mapping used for readable debug outputs. Features that are already
    # unitless or direction-like are returned unchanged.
    if name == "soc":
        return value * 100.0
    if name == "remaining_time":
        return value * EPISODE_STEPS
    if name == "needed_soc":
        return value * 100.0
    if name == "battery_capacity_kwh":
        return value * float(EV_CAPACITY_OBS_SCALE_KWH)
    if name == "max_power_kw":
        return value * float(EV_CHARGER_POWER_OBS_SCALE_KW)
    if name == "switch_count":
        return value * max(float(MAX_SWITCH_COUNT), 1.0)
    return value


def normalize_observation(obs):
    """
    Normalize all stations in a raw observation.

    The returned tensor keeps the original station/feature layout; only numeric
    scale changes. This is the main entry point used by training and evaluation.
    """
    obs = _to_tensor(obs)
    normalized_obs = obs.clone()
    ev_block_dim = EV_FEAT_DIM * MAX_EV_PER_STATION

    state_dim = obs.shape[-1]
    ev_block_end = min(ev_block_dim, state_dim)
    num_evs = ev_block_end // EV_FEAT_DIM

    if num_evs > 0:
        ev_slice = slice(0, num_evs * EV_FEAT_DIM)
        raw_ev = obs[:, ev_slice].reshape(obs.shape[0], num_evs, EV_FEAT_DIM)
        norm_ev = normalized_obs[:, ev_slice].reshape(obs.shape[0], num_evs, EV_FEAT_DIM)

        for feat_offset, feat_name in enumerate(EV_FEATURE_NAMES):
            if feat_offset >= EV_FEAT_DIM:
                break
            values = raw_ev[..., feat_offset]
            if feat_name == "presence":
                norm_ev[..., feat_offset] = values
            elif feat_name == "soc":
                norm_ev[..., feat_offset] = values / 100.0
            elif feat_name == "remaining_time":
                norm_ev[..., feat_offset] = values / EPISODE_STEPS
            elif feat_name == "needed_soc":
                norm_ev[..., feat_offset] = torch.clamp(values / 100.0, -1.0, 1.0)
            elif feat_name == "battery_capacity_kwh":
                norm_ev[..., feat_offset] = torch.clamp(
                    values / max(float(EV_CAPACITY_OBS_SCALE_KWH), 1.0), 0.0, 1.0
                )
            elif feat_name == "max_power_kw":
                norm_ev[..., feat_offset] = torch.clamp(
                    values / max(float(EV_CHARGER_POWER_OBS_SCALE_KW), 1.0), 0.0, 1.0
                )
            elif feat_name == "switch_count":
                norm_ev[..., feat_offset] = torch.clamp(
                    values / max(float(MAX_SWITCH_COUNT), 1.0), 0.0, 1.0
                )
            elif feat_name == "last_direction":
                norm_ev[..., feat_offset] = torch.clamp(values, -1.0, 1.0)

    tail_idx = ev_block_end
    if LOCAL_DEMAND_STEPS > 0 and tail_idx < state_dim:
        # The demand lookahead tail is normalized against the full observed
        # AG-request range, so positive and negative requests are balanced
        # around zero for the networks.
        lookahead = int(LOCAL_DEMAND_STEPS)
        end_ag = min(tail_idx + lookahead, state_dim)
        if end_ag > tail_idx:
            ag_slice = slice(tail_idx, end_ag)
            normalized_obs[:, ag_slice] = torch.clamp(
                (obs[:, ag_slice] - AG_REQUEST_CENTER) / AG_REQUEST_SCALE,
                -float(OBS_DEMAND_CLAMP),
                float(OBS_DEMAND_CLAMP),
            )
            tail_idx = end_ag

    if LOCAL_USE_TRACKING_ENABLED and tail_idx < state_dim:
        normalized_obs[:, tail_idx] = torch.clamp(
            obs[:, tail_idx], 0.0, 1.0
        )
        tail_idx += 1

    if LOCAL_USE_FLEET_RESIDUAL and tail_idx < state_dim:
        # Same units and same divisor as ``demand_0``: the day's own
        # instruction envelope, so the two are directly comparable.
        normalized_obs[:, tail_idx] = torch.clamp(
            obs[:, tail_idx] / AG_REQUEST_SCALE,
            -float(OBS_DEMAND_CLAMP),
            float(OBS_DEMAND_CLAMP),
        )
        tail_idx += 1

    if LOWER_BID_CONTEXT_USE_OBS and tail_idx < state_dim:
        for feature_name in BID_CONTEXT_FEATURES:
            if tail_idx >= state_dim:
                break
            if feature_name == "instruction_scale_kw":
                normalized_obs[:, tail_idx] = torch.clamp(
                    obs[:, tail_idx] / _market_context_scale(feature_name),
                    0.0,
                    float(OBS_DEMAND_CLAMP),
                )
            else:
                normalized_obs[:, tail_idx] = torch.clamp(
                    obs[:, tail_idx], -1.0, 1.0
                )
            tail_idx += 1

    if BESS_CONTEXT_USE_OBS and tail_idx < state_dim:
        for feature_name in BESS_CONTEXT_FEATURES:
            if tail_idx >= state_dim:
                break
            normalized_obs[:, tail_idx] = _normalize_bess_context(
                feature_name, obs[:, tail_idx]
            )
            tail_idx += 1

    if LOCAL_USE_STEP and tail_idx < state_dim:
        # Current time is represented as episode progress. Clamp prevents a
        # malformed step index from leaking out-of-range values.
        normalized_obs[:, tail_idx] = torch.clamp(
            obs[:, tail_idx] / EPISODE_STEPS, 0.0, 1.0
        )

    return normalized_obs


def denormalize_soc(normalized_soc):
    return normalized_soc * 100.0


def normalize_ag_request(raw_ag):
    return torch.clamp(
        (raw_ag - AG_REQUEST_CENTER) / AG_REQUEST_SCALE,
        -float(OBS_DEMAND_CLAMP),
        float(OBS_DEMAND_CLAMP),
    )


def denormalize_ev_capacity_kwh(normalized_capacity):
    return normalized_capacity * float(EV_CAPACITY_OBS_SCALE_KWH)


def denormalize_ev_max_power_kw(normalized_max_power):
    return normalized_max_power * float(EV_CHARGER_POWER_OBS_SCALE_KW)
