"""未見指令の合格率を、同じ360組（検証日5日 × 未見指令24本 × EV実現3本）で方法ごとに測るための共通設定。

marl_unseen_aemo_ab2200_20261002 と同じコード（学習時のコードに、学習で引いた指令を
除く3ファイルを重ねたもの）と同じ実行時設定を使う。指令の引き方、EV実現の seed は
tools/evaluate_final_system_on_bid_bank.py の main() と同じ式にする。
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MARL_RUN = ROOT / 'execute_results/marl_unseen_aemo_ab2200_20261002'
PINNED = ROOT / 'execute_results/ab_resume_2500_20260930/runtime_source'
SOURCE = MARL_RUN / 'runtime_source'
MODEL = ROOT / 'archive/prod_aemoplan_AB_7station_20260926_210314'
BANK = ROOT / 'execute_results/bid_banks/validation_5_minmedmax_3of128ev_128cmd_all_commands_aemo_plan_deviation'
LIBRARY = ROOT / 'data/aemo/nem/command_libraries/plan_deviation_calendar_day'
OVERLAY = (
    'market/activation_scenarios.py',
    'training/lower_bid_training.py',
    'tools/evaluate_final_system_on_bid_bank.py',
)
EPISODE = 2200
COMMANDS_PER_DAY = 24
EV_SEEDS = 3
BASE_SEED = 1_422_090
SEEN_CACHE = HERE / 'seen_commands.json'


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure() -> dict:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['MPLBACKEND'] = 'Agg'
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '1'
    for key in list(os.environ):
        if key.startswith('EVMA_'):
            del os.environ[key]
    os.environ['EVMA_ACTIVATION_SIGNAL_SET'] = 'aemo_plan_deviation'

    context = json.loads((MODEL / 'resume/latest.json').read_text(encoding='utf-8'))['context']
    digest = hashlib.sha256()
    for relative in sorted(context['source']['files']):
        digest.update(relative.encode('utf-8'))
        digest.update(b'\0')
        digest.update((PINNED / relative).read_bytes())
        digest.update(b'\0')
    assert digest.hexdigest() == context['source']['sha256'], '学習時のコードと一致しない'
    for relative in sorted(context['source']['files']):
        if relative not in OVERLAY:
            assert _sha(SOURCE / relative) == _sha(PINNED / relative), relative
    manifest = json.loads((BANK / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['complete'] and manifest['completed_days'] == 5

    sys.path.insert(0, str(SOURCE))
    os.chdir(ROOT)
    overrides = {
        'PROJECT_ROOT': str(SOURCE),
        'ACTIVATION_SIGNAL_SET': 'aemo_plan_deviation',
        'ACTIVATION_SCENARIO_DIR': str(LIBRARY),
        'LOWER_TRAIN_UPPER_BID_ACTIVATION_SOURCE_DIR': str(LIBRARY),
        'LOWER_TRAIN_UPPER_BID_TEST_BANK_DIR': str(BANK),
    }
    for name in ('EnvConfig', 'Config', 'environment.observation_config'):
        module = importlib.import_module(name)
        assert Path(module.__file__).resolve().is_relative_to(SOURCE)
        for key, value in context['runtime'][name.rsplit('.', 1)[-1]].items():
            if isinstance(value, list) and key.endswith(('FEATURES', 'FEATURE_NAMES')):
                value = tuple(value)
            setattr(module, key, value)
        if name in ('EnvConfig', 'Config'):
            for key, value in overrides.items():
                setattr(module, key, value)
    config = importlib.import_module('Config')
    assert config.NUM_STATIONS == 7 and config.Q_MIX_GLOBAL_WEIGHT == 0.5

    activation = importlib.import_module('market.activation_scenarios')
    original_signature = activation.activation_library_signature
    signature = original_signature(LIBRARY)
    assert signature['activation_library_sha256'] == manifest['settings']['activation_library_sha256']

    def fixed_signature(directory=None):
        path = Path(directory or LIBRARY).resolve()
        if path != LIBRARY.resolve():
            return original_signature(directory)
        return dict(signature)

    activation.activation_library_signature = fixed_signature
    return {'context': context, 'pinned_source_sha256': digest.hexdigest()}


def seen_commands() -> set[str]:
    if SEEN_CACHE.exists():
        return set(json.loads(SEEN_CACHE.read_text(encoding='utf-8')))
    evaluator = importlib.import_module('tools.evaluate_final_system_on_bid_bank')
    seen = evaluator._commands_seen_in_training(MODEL)
    SEEN_CACHE.write_text(json.dumps(sorted(seen), ensure_ascii=False) + '\n', encoding='utf-8')
    return seen


def day_cases() -> list[tuple[int, dict, dict]]:
    """(day_index, entry, fixed_bid with the 24 unseen commands) in evaluator order."""
    from training.bid_bank import BidBank
    from training.lower_bid_training import _activation_scenarios_for_day

    seen = seen_commands()
    assert len(seen) == 1785, len(seen)
    bank = BidBank(BANK)
    out = []
    for day_index, entry in enumerate(list(bank.entries)):
        fixed_bid = dict(bank.load_entry(entry))
        payloads, mode = _activation_scenarios_for_day(
            fixed_bid.get('service_date'),
            int(fixed_bid.get('forecast_seed', 0)) + 512_209,
            n_scenarios=COMMANDS_PER_DAY,
            scenario_partition='holdout',
            exclude_sources=seen,
        )
        fixed_bid['activation_scenario_payload'] = list(payloads)
        fixed_bid['activation_scenarios'] = len(payloads)
        fixed_bid['activation_mode'] = mode
        out.append((day_index, entry, fixed_bid))
    return out


def day_base_seed(day_index: int) -> int:
    return BASE_SEED + day_index * 100_000


def realized_seed(day_index: int, scenario: int, k: int) -> int:
    # training/evaluate_controller_precision.py: base_seed + 1000 * s + k
    return day_base_seed(day_index) + 1000 * scenario + k
