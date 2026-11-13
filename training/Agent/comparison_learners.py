"""Learners added for the algorithm comparison (2026-10-29), chosen by Config.MARL_ALGORITHM.

standard_maddpg.build_marl_agent knows 'hybrid' (the research learner) and
'maddpg' (Lowe et al. 2017). build_learner adds

  maddpg_per_ev  MADDPG with one actor per EV slot (training/Agent/per_ev_maddpg.py)
  maac           MADDPG with MAAC's attention critic (training/Agent/maac.py)
  mappo          MAPPO on the station actors (training/Agent/mappo.py)

and passes every other name to build_marl_agent, so 'hybrid' and 'maddpg' are
built exactly as before.

training/train.py and training/Agent/standard_maddpg.py are among the files the
training-resume guard fingerprints (training/training_resume.py), so editing
them would refuse to resume every run started before the edit. They are left
as they are: training/system_controller.py (evaluation) calls build_learner,
and a training entry point calls install() before it imports the trainer, which
points standard_maddpg.build_marl_agent at build_learner for that process.
"""
from __future__ import annotations

from . import standard_maddpg

_build_marl_agent = standard_maddpg.build_marl_agent

COMPARISON_ALGORITHMS = ("maddpg_per_ev", "maac", "mappo")


def build_learner(s_dim, max_evs_per_station, n_agent, **kwargs):
    """The learner named by Config.MARL_ALGORITHM, including the comparison learners."""
    from Config import MARL_ALGORITHM

    if MARL_ALGORITHM == "maddpg_per_ev":
        from .per_ev_maddpg import PerEVMADDPG
        return PerEVMADDPG(s_dim, max_evs_per_station, n_agent, **kwargs)
    if MARL_ALGORITHM == "maac":
        from .maac import MAACMADDPG
        return MAACMADDPG(s_dim, max_evs_per_station, n_agent, **kwargs)
    if MARL_ALGORITHM == "mappo":
        from .mappo import MAPPO
        return MAPPO(s_dim, max_evs_per_station, n_agent, **kwargs)
    return _build_marl_agent(s_dim, max_evs_per_station, n_agent, **kwargs)


def install() -> None:
    """Make standard_maddpg.build_marl_agent build the comparison learners too (this process only)."""
    standard_maddpg.build_marl_agent = build_learner
