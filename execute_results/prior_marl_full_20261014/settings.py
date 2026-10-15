"""Process-local configuration; main configuration files are read-only."""
from pathlib import Path
import importlib.util

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('afm_main_settings', ROOT / 'execute_results/resbid_6station_20261012/stage_worker.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def configure(task='aemo'):
    config = base.configure(task)
    assert config.Q_MIX_GLOBAL_WEIGHT == .5
    import torch
    torch.set_num_threads(1)
    return config


def banks(task='aemo'):
    return base.banks(task)
