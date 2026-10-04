"""What the local and global critics tell each EV's action, per run, on the same states.

usage: critic_probe.py [EPISODE]   (default 400; run with the launcher's environment variables)

Loads the actors and critics each run saved at interim test EPISODE (AB on the
AEMO plan-deviation bank, and the same with per-EV local critics), and 4096
states from the last 20,000 transitions of the per-EV run's replay at its
newest exact-resume checkpoint. At each state every station's own
actor proposes its actions, and the actor-update pipeline (rating scaling, SoC
limits with the straight-through estimator) is rebuilt to take the gradient of
the local and the global Q with respect to each EV's normalized action: the
signal the actor receives for that EV before it reaches the actor's weights.

EVs are grouped by their need at the state: at or above target, or below target
with laxity (steps left minus the steps full-power charging needs) above 48,
between 0 and 48, or at most 0. A positive gradient asks the EV to charge more.
Also printed: the per-EV local rewards stored in replay, split into departure
steps and the rest.
"""
from __future__ import annotations

import glob
import os
import sys

import torch

ROOT = r"C:\Users\admin\Desktop\EVMALOCALUPDATE"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import training.Agent.maddpg as m  # noqa: E402
from Config import EPISODE_STEPS, MAX_EV_PER_STATION, POWER_TO_ENERGY  # noqa: E402
from environment.observation_config import local_obs_dim  # noqa: E402

EP = int((sys.argv[1:] or ["400"])[0])
RUNS = {
    "AB": (os.path.join(ROOT, "archive", "prod_aemoplan_AB_7station_20260926_210314"), False),
    "per-EV": (sorted(glob.glob(os.path.join(ROOT, "archive", "prod_aemoplan_AB_perev_7station_*")))[-1], True),
}
dev = m.device
N, E = 7, MAX_EV_PER_STATION


def agent_for(run: str, per_ev: bool) -> m.MADDPG:
    m.LOCAL_CRITIC_PER_EV = per_ev
    agent = m.MADDPG(s_dim=local_obs_dim(E), max_evs_per_station=E, n_agent=N)
    test_dir = os.path.join(run, "results", f"TEST{EP}")
    agent.load_actors(test_dir, EP, map_location=dev)
    bundle = torch.load(os.path.join(test_dir, f"agent_state_ep{EP}.pth"), map_location=dev, weights_only=False)
    for mods, key in ((agent.critics, "critics"), (agent.critics2, "critics2")):
        for mod, sd in zip(mods, bundle[key]):
            mod.load_state_dict(sd)
    agent.global_critic1.load_state_dict(bundle["global_critic1"])
    agent.global_critic2.load_state_dict(bundle["global_critic2"])
    agent.set_test_mode(True)
    return agent


def action_gradients(agent: m.MADDPG, s: torch.Tensor):
    B = s.size(0)
    zeros = torch.zeros((B, N), device=dev)
    ctx = agent._build_update_ctx(s, s, torch.zeros((B, N, E), device=dev), zeros, zeros, zeros, None)
    with torch.no_grad():
        proposal = torch.stack([agent.actors[i](s[:, i, :]) for i in range(N)], dim=1)
    cur = proposal.clone().requires_grad_(True)
    actions = (cur * ctx["max_power_kw"]).masked_fill(ctx["ev_padding_mask"], 0.0)
    clamped, station_kw = agent._apply_soc_constraint(
        actions, ctx["current_socs"], ctx["ev_padding_mask"], use_ste=True,
        capacity_kwh=ctx["capacity_kwh"], max_power_kw=ctx["max_power_kw"],
        soc_step_per_kw=ctx["soc_step_per_kw"], kw_per_soc_step=ctx["kw_per_soc_step"],
    )
    station_norm = torch.clamp(station_kw / ctx["max_station_power"], -1.0, 1.0)
    q_local = 0.0
    for i in range(N):
        a_i = torch.clamp(clamped[:, i, :] * ctx["inv_max_power_kw"][:, i, :], -1.0, 1.0)
        q_local = q_local + agent.critics[i](s[:, i, :], a_i, actual_station_powers=station_norm[:, i]).sum()
    g_local = torch.autograd.grad(q_local, cur, retain_graph=True)[0]
    s_global = agent._convert_to_global_critic_obs(s, station_kw)
    a_norm = agent._normalize_power_by_limit(clamped.masked_fill(ctx["ev_padding_mask"], 0.0),
                                             ctx["max_power_kw"], ctx["inv_max_power_kw"])
    q1, _ = agent.global_critic1(s_global, a_norm, ctx["key_padding_mask"], actual_station_powers=station_norm)
    q2, _ = agent.global_critic2(s_global, a_norm, ctx["key_padding_mask"], actual_station_powers=station_norm)
    g_global = torch.autograd.grad(torch.minimum(q1, q2).sum(), cur)[0]
    return proposal, g_local, g_global, ctx


def weight_space(agent: m.MADDPG, s: torch.Tensor, g_local: torch.Tensor, g_global: torch.Tensor):
    """Per station: the two sources' gradient norms on the actor's weights, as the learner logs them.

    The actor update backpropagates the batch-mean Q, so the weight gradient is
    the action gradient divided by the batch size and pulled through the actor.
    """
    B = s.size(0)
    out = []
    for i in range(N):
        params = list(agent.actors[i].parameters())
        a = agent.actors[i](s[:, i, :])
        norms = []
        for g in (g_local, g_global):
            grads = torch.autograd.grad(a, params, grad_outputs=g[:, i, :] / B, retain_graph=True, allow_unused=True)
            norms.append(float(torch.sqrt(sum((x * x).sum() for x in grads if x is not None))))
        out.append(norms)
    return torch.tensor(out)


def groups(ctx):
    block = ctx["ev_block"]
    need_pct = block[..., m.NEEDED_FEATURE_IDX] * 100.0
    remaining = torch.round(block[..., m.REMAINING_FEATURE_IDX] * float(EPISODE_STEPS))
    full_step_pct = ctx["max_power_kw"] * float(POWER_TO_ENERGY) * 100.0 / ctx["capacity_kwh"]
    laxity = remaining - torch.clamp(need_pct, min=0.0) / torch.clamp(full_step_pct, min=1e-6)
    present = ~ctx["ev_padding_mask"]
    return {
        "at/above target": present & (need_pct <= 0),
        "below, laxity > 48": present & (need_pct > 0) & (laxity > 48),
        "below, laxity 0-48": present & (need_pct > 0) & (laxity > 0) & (laxity <= 48),
        "below, laxity <= 0": present & (need_pct > 0) & (laxity <= 0),
    }, laxity, present


def main() -> None:
    per_ev_run = RUNS["per-EV"][0]
    # Exact-resume states rotate; take the newest (its replay holds every
    # transition up to it, and the probe uses the last 20,000).
    newest = max(glob.glob(os.path.join(per_ev_run, "resume", "training_state_ep*.pth")),
                 key=lambda p: int(p.rsplit("ep", 1)[1].split(".")[0]))
    print(f"states from {os.path.basename(newest)}")
    state = torch.load(newest, map_location="cpu", weights_only=False)
    replay = state["agent"]["replay"]
    ptr = int(replay["ptr"])
    tensors = replay["tensors"]
    gen = torch.Generator().manual_seed(0)
    idx = torch.randint(max(0, ptr - 20000), ptr, (4096,), generator=gen)
    s = tensors["s"][idx].to(device=dev, dtype=torch.float32)
    r_ev, nxt = tensors["r_ev_local"], tensors["next_slot"]
    recent = slice(max(0, ptr - 20000), ptr)
    left = (nxt[recent] < 0) & (tensors["s"][recent][..., 0:E * 6:6] > 0.5)
    stay = (nxt[recent] >= 0)
    print(f"replay: {ptr} transitions; probe states from the last 20,000")
    print(f"per-EV local reward on departure steps: mean {r_ev[recent][left].mean():+.3f}  "
          f"|mean| {r_ev[recent][left].abs().mean():.3f}  n {int(left.sum())}")
    print(f"per-EV local reward on other steps:     mean {r_ev[recent][stay].mean():+.5f}  "
          f"|mean| {r_ev[recent][stay].abs().mean():.5f}  n {int(stay.sum())}")
    del state, tensors

    for name, (run, per_ev) in RUNS.items():
        agent = agent_for(run, per_ev)
        proposal, g_local, g_global, ctx = action_gradients(agent, s)
        masks, laxity, present = groups(ctx)
        mixed = 0.5 * g_local + 0.5 * g_global
        print(f"== {name} (ep{EP})  EVs {int(present.sum())}")
        print(f"   {'group':20s} {'n':>6s} {'action':>7s} {'dQloc/da':>9s} {'dQglob/da':>10s} {'mixed>0':>8s}")
        for label, mask in masks.items():
            if int(mask.sum()) == 0:
                continue
            print(f"   {label:20s} {int(mask.sum()):6d} {proposal[mask].mean():+7.3f} {g_local[mask].mean():+9.4f} "
                  f"{g_global[mask].mean():+10.4f} {(mixed[mask] > 0).float().mean():8.2f}")
        # Within a station: does the local gradient rank the EVs by urgency?
        corr = []
        for b in range(s.size(0)):
            for i in range(N):
                p = present[b, i]
                if int(p.sum()) < 3:
                    continue
                x, y = -laxity[b, i][p], g_local[b, i][p]
                if float(x.std()) > 0 and float(y.std()) > 0:
                    corr.append(float(torch.corrcoef(torch.stack([x, y]))[0, 1]))
        corr_t = torch.tensor(corr)
        print(f"   within-station corr(urgency, dQloc/da): mean {corr_t.mean():+.3f} "
              f"(share > 0: {(corr_t > 0).float().mean():.2f}, station-states {len(corr)})")
        spread = []
        for b in range(s.size(0)):
            for i in range(N):
                p = present[b, i]
                if int(p.sum()) >= 3:
                    spread.append((float(g_local[b, i][p].std()), float(g_global[b, i][p].std()),
                                   float(g_local[b, i][p].abs().mean()), float(g_global[b, i][p].abs().mean())))
        sp = torch.tensor(spread)
        print(f"   within-station spread over EVs: local sd {sp[:, 0].mean():.4f} (|mean| {sp[:, 2].mean():.4f})  "
              f"global sd {sp[:, 1].mean():.4f} (|mean| {sp[:, 3].mean():.4f})")
        p = present
        print(f"   action space: |g| local {g_local[p].abs().mean():.4f} global {g_global[p].abs().mean():.4f} "
              f"(global/local {g_global[p].abs().mean() / g_local[p].abs().mean():.1f});  "
              f"|mean g|/mean|g| local {g_local[p].mean().abs() / g_local[p].abs().mean():.2f} "
              f"global {g_global[p].mean().abs() / g_global[p].abs().mean():.2f}")
        w = weight_space(agent, s, g_local, g_global)
        print(f"   weight space (as logged): local {w[:, 0].mean():.3f} global {w[:, 1].mean():.3f} "
              f"(global/local {(w[:, 1] / w[:, 0]).mean():.2f})")


if __name__ == "__main__":
    main()
