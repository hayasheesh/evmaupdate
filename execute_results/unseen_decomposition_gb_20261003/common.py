"""GB（Elexon、計画からのずれ）の未見指令の組で、GBで学習したAB・中央LP・ルールベース・完全情報LPを比べるための共通設定。

組は GB の検証日5日 × 未見指令24本 × EV実現3本の360組。検証日とEV実現の seed は AEMO・ERCOT の比較
（unseen_decomposition_aemo_20261002 / unseen_decomposition_ercot_20261002）と同じで、違うのは指令と入札だけ。
指令は GB の holdout から、GB のモデルが2000回の学習を終えるまでに引く指令をすべて除いて引く。
引く指令は seed から決まるので、学習が終わる前でも2000回分（環境のエピソード2025回分）を正確に作れる
（training.lower_bid_training.lower_commands_seen_in_pretrain）。こうしておけば、途中のチェックポイントにも
2000回のモデルにも未見の組になる。
コードは AEMO・ERCOT の比較と同じもの（学習時のコードに、学習で引いた指令を除く3ファイルを重ねたもの）。
3つのモデルの source sha256 は同じ。実行時設定は GB のモデルのもの。
指令ライブラリ（4,305本）の内容は、入札バンクに記録された署名をそのまま使う。ファイル数だけ照合する。
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
MODEL = ROOT / 'archive/prod_elexonplan_AB_7station_20261002_215844'
FINAL_TRAINING_EPISODES = 2000
WARMUP_ENVIRONMENT_EPISODES = 25
MODELS = {
    'gb': (MODEL, int(os.environ.get('GB_MODEL_EPISODE', '1240'))),
}
BANK = ROOT / 'execute_results/bid_banks/validation_5_minmedmax_3of128ev_128cmd_all_commands_elexon_plan_deviation'
LIBRARY = ROOT / 'data/elexon/command_libraries/battery_plan_deviation_calendar_day'
OVERLAY = (
    'market/activation_scenarios.py',
    'training/lower_bid_training.py',
    'tools/evaluate_final_system_on_bid_bank.py',
)
COMMANDS_PER_DAY = 24
EV_SEEDS = 3
BASE_SEED = 1_422_090
SEEN_CACHE = HERE / 'seen_commands_gb_to_ep2000.json'
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
    os.environ['EVMA_ACTIVATION_SIGNAL_SET'] = 'elexon_plan_deviation'

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
    assert int(manifest['completed_environment_episodes']) - int(manifest['completed_training_episode']) == WARMUP_ENVIRONMENT_EPISODES
    context = manifest['context']
    runtime = context['runtime']
    seen = lower_commands_seen_in_pretrain(
        list(BidBank(context['train_bid_bank']['path']).entries),
        list(BidBank(context['test_bid_bank']['path']).entries),
        environment_episodes=FINAL_TRAINING_EPISODES + WARMUP_ENVIRONMENT_EPISODES,
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
