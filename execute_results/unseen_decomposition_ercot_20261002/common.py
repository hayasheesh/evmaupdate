"""ERCOTの未見指令の組で、AEMOで学習したAB・ERCOTで学習したAB・完全情報LPを比べるための共通設定。

組は ERCOT の検証日5日 × 未見指令24本 × EV実現3本の360組。指令は ERCOT の holdout から、
ERCOT のモデルが学習中に引いた指令（ep1760 を保存した時点まで）を除いて引く。AEMO のモデルは
ERCOT の指令を一度も見ていないので、どちらのモデルにとっても未見になる。
コードは2つのモデルに共通（source sha256 が同じ）で、unseen_decomposition_aemo_20261002 と同じもの
（学習時のコードに、学習で引いた指令を除く3ファイルを重ねたもの）を使う。実行時設定は ERCOT の
モデルのもの。AEMO のモデルとの違いは市場（指令ライブラリと入札バンク）の指定だけ。
指令ライブラリ（46,994本）の内容は、入札バンクに記録された署名をそのまま使い、毎回の
全ファイルのハッシュ計算は省く（ercot_ab_fast_resume_20261001/fast_resume.py と同じ扱い）。ファイル数だけ照合する。
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
MODEL = ROOT / 'archive/prod_ercotplan_AB_7station_20260930_221551'
MODELS = {
    'aemo': (ROOT / 'archive/prod_aemoplan_AB_7station_20260926_210314', 2200),
    'ercot': (MODEL, 1760),
}
BANK = ROOT / 'execute_results/bid_banks/validation_5_minmedmax_3of128ev_128cmd_all_commands_ercot_plan_deviation'
LIBRARY = ROOT / 'data/ercot/sced/command_libraries/esr_plan_deviation_calendar_day'
OVERLAY = (
    'market/activation_scenarios.py',
    'training/lower_bid_training.py',
    'tools/evaluate_final_system_on_bid_bank.py',
)
COMMANDS_PER_DAY = 24
EV_SEEDS = 3
BASE_SEED = 1_422_090
SEEN_CACHE = HERE / 'seen_commands_ercot.json'
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
    os.environ['EVMA_ACTIVATION_SIGNAL_SET'] = 'ercot_plan_deviation'

    context = json.loads((MODEL / 'resume/latest.json').read_text(encoding='utf-8'))['context']
    aemo_context = json.loads((MODELS['aemo'][0] / 'resume/latest.json').read_text(encoding='utf-8'))['context']
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
    assert Path(signature['activation_source_dir']).resolve() == LIBRARY.resolve()
    assert sum(1 for p in LIBRARY.iterdir() if p.suffix == '.csv') == int(signature['activation_library_file_count'])

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
    config = importlib.import_module('Config')
    assert config.NUM_STATIONS == 7 and config.Q_MIX_GLOBAL_WEIGHT == 0.5
    assert Path(config.LOWER_TRAIN_UPPER_BID_ACTIVATION_SOURCE_DIR).resolve() == LIBRARY.resolve()

    activation = importlib.import_module('market.activation_scenarios')
    lower_bid = importlib.import_module('training.lower_bid_training')

    def recorded_signature(directory=None):
        requested = Path(directory or LIBRARY).resolve()
        if requested != LIBRARY.resolve():
            raise ValueError(f'unexpected command source: {requested}')
        return dict(signature)

    activation.activation_library_signature = recorded_signature
    if hasattr(lower_bid, 'activation_library_signature'):
        lower_bid.activation_library_signature = recorded_signature
    return {'context': context, 'pinned_source_sha256': digest.hexdigest(), 'signature': signature}


def seen_commands() -> set[str]:
    if SEEN_CACHE.exists():
        return set(json.loads(SEEN_CACHE.read_text(encoding='utf-8')))
    from training.bid_bank import BidBank
    from training.lower_bid_training import lower_commands_seen_in_pretrain

    manifest = json.loads((MODEL / 'resume/latest.json').read_text(encoding='utf-8'))
    # 学習は続いているので、保存が ep1760 より後なら除く指令は多めになる（ep1760 にとっても未見のまま）
    assert int(manifest['completed_training_episode']) >= MODELS['ercot'][1], manifest['completed_training_episode']
    context = manifest['context']
    runtime = context['runtime']
    seen = lower_commands_seen_in_pretrain(
        list(BidBank(context['train_bid_bank']['path']).entries),
        list(BidBank(context['test_bid_bank']['path']).entries),
        environment_episodes=int(manifest['completed_environment_episodes']),
        interim_test_episodes=int(runtime['Config']['INTERIM_TEST_EPISODES']),
        library_dir=LIBRARY,
    )
    SEEN_CACHE.write_text(json.dumps(sorted(seen), ensure_ascii=False) + '\n', encoding='utf-8')
    return seen


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
