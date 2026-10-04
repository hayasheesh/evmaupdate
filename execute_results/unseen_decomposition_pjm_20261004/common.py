"""PJM RegD の実指令で、ルールベース・完全情報LP（あとで GB と同じく MARL も）を比べるための共通設定。

組は PJM の検証日5日 × 未見指令24本 × EV実現3本の360組。検証日と EV 実現の seed は AEMO・ERCOT・GB の比較と同じで、
違うのは指令と入札だけ。入札（validation_5 の PJM バンク）は、疑似指令 pjm_regd_phase_shift（実指令の train の日だけ
から作ったもの）で作った。評価で当てる指令は、実指令 pjm_regd の holdout（test の日）から引く。学習も疑似指令しか
引いていないので、実指令の test の日はすべて未見になる（除く指令は無い）。
コードは AEMO・ERCOT・GB の比較と同じもの。実行時設定は PJM のモデル（prod_pjmregd_AB_7station）のもので、
評価で引く指令の置き場所（LOWER_TRAIN_UPPER_BID_ACTIVATION_SOURCE_DIR）だけを実指令に替える。
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
PINNED = ROOT / 'execute_results/ab_resume_2500_20260930/runtime_source'
SOURCE = ROOT / 'execute_results/marl_unseen_aemo_ab2200_20261002/runtime_source'
MODEL = ROOT / 'archive/prod_pjmregd_AB_7station_20261004_044902'
MODELS = {
    'pjm': (MODEL, int(os.environ.get('PJM_MODEL_EPISODE', '200'))),
}
BANK = ROOT / 'execute_results/bid_banks/validation_5_minmedmax_3of128ev_128cmd_all_commands_pjm_regd_phase_shift'
LIBRARY = ROOT / 'data/pjm/command_libraries/regd_calendar_day'
BANK_LIBRARY = ROOT / 'data/pjm/command_libraries/regd_phase_shift_train_calendar_day'
OVERLAY = (
    'market/activation_scenarios.py',
    'training/lower_bid_training.py',
    'tools/evaluate_final_system_on_bid_bank.py',
)
COMMANDS_PER_DAY = 24
EV_SEEDS = 3
BASE_SEED = 1_422_090
SIGNATURE_KEYS = (
    'activation_signal_mode', 'activation_signal_regime',
    'activation_source_dir', 'activation_library_file_count',
    'activation_library_sha256',
)


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
    aemo_context = json.loads((ROOT / 'archive/prod_aemoplan_AB_7station_20260926_210314/resume/latest.json').read_text(encoding='utf-8'))['context']
    assert context['source']['sha256'] == aemo_context['source']['sha256']
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
    signature = {key: manifest['settings'][key] for key in SIGNATURE_KEYS}
    assert Path(signature['activation_source_dir']).resolve() == BANK_LIBRARY.resolve()

    sys.path.insert(0, str(SOURCE))
    os.chdir(ROOT)
    for name in ('EnvConfig', 'Config', 'environment.observation_config'):
        module = importlib.import_module(name)
        assert Path(module.__file__).resolve().is_relative_to(SOURCE)
        for key, value in context['runtime'][name.rsplit('.', 1)[-1]].items():
            if isinstance(value, list) and key.endswith(('FEATURES', 'FEATURE_NAMES')):
                value = tuple(value)
            setattr(module, key, value)
        if name in ('EnvConfig', 'Config'):
            setattr(module, 'PROJECT_ROOT', str(SOURCE))
            setattr(module, 'LOWER_TRAIN_UPPER_BID_TEST_BANK_DIR', str(BANK))
            # 評価で当てる指令は実指令から引く
            setattr(module, 'LOWER_TRAIN_UPPER_BID_ACTIVATION_SOURCE_DIR', str(LIBRARY))
            setattr(module, 'ACTIVATION_SCENARIO_DIR', str(LIBRARY))
    config = importlib.import_module('Config')
    assert config.NUM_STATIONS == 7 and config.Q_MIX_GLOBAL_WEIGHT == 0.5

    lower_bid = importlib.import_module('training.lower_bid_training')
    assert Path(lower_bid.LOWER_TRAIN_UPPER_BID_ACTIVATION_SOURCE_DIR).resolve() == LIBRARY.resolve()
    return {'context': context, 'pinned_source_sha256': digest.hexdigest(), 'signature': signature}


def seen_commands() -> set[str]:
    """学習は疑似指令（別のライブラリ）しか引いていないので、実指令で除くものは無い。"""
    return set()


def day_cases() -> list[tuple[int, dict, dict]]:
    from training.bid_bank import BidBank
    from training.lower_bid_training import _activation_scenarios_for_day

    seen = seen_commands()
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
    return day_base_seed(day_index) + 1000 * scenario + k
