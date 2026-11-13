"""実データの1日を中心にした世界で、行動器を学習する（AEMO 6 station、Windows と研究室、2026-10-19）。

    python realft_worker.py <mode> <preflight|pretrain> [--episodes 1000]

学習の世界：その日の入札だけの束（build_real.bank_name）では、学習の reset は入札の到着の確率（その日の件数から作った値）で
くじを引き、0時のEVは前の2日の確率で流す（realday.install_preroll）。中間テストも同じ束と同じ世界。本当の状況（その日の実際の
セッション）は、どの mode でも学習にも中間テストにも使わない。観測の正規化は、どの mode も本線の実行のものを使う。

  busy / quiet / average     本線の保存した状態（2000回目）から、その日の世界で続ける（finetune）。
  control                    本線の保存した状態から、本線の世界と本線の束で続ける（学習を足しただけの効果と分けるため）。
  scratch_<busy|quiet|average>  その日の世界で一から学習する。探索を弱めていく予定（epsilon、OU のノイズ）は、本線の2000回用の
                             半分（epsilon は500回、OU は750回で最後の値）にして、1000回で本線と同じ段階まで進むようにする。
  multi                      いろいろな実データの日（build_multi.py の束、学習用25日・中間テスト用5日）で一から2000回学習する。
                             0時のEVは、reset に渡された到着の確率から、その日の前の2日を選んで流す。探索の予定は本線と同じ。
  multift_<busy|quiet|average>  multi の保存した状態（2000回目）から、その日の世界で続ける（finetune）。
  fc_<busy|quiet|average>    本線の保存した状態から、前日の昼の予測（直近4週の平日の平均、realday.forecast_profiles）で作った
                             入札と学習の世界で続ける。本当の状況は同じ日。
  ml_<busy|quiet|average>    同じく、機械学習の水準の予測（realday.ALPHA_ML、外れが単純な予測の0.7倍）で続ける。
  scratch_<世界>、multift_<世界>  <世界> は busy・fc_busy・ml_busy など。その世界で一から、または multi から続ける。
保存した状態から続ける mode は、行動器・批評家・目標のネットワーク、最適化の状態、回数、リプレイをそのまま読み込む
（agent.load_training_resume_state_dict）。リプレイを空にして再開すると、本線の世界のままでも行動器が揺れて悪くなったため。
回数の数えは、train.py が学習の始めに0に戻すので、エピソードの始めに2000を足し直す（探索の大きさは2001回目以降と同じ）。
共有のコードは変えず、このプロセスの中だけで training.train.build_marl_agent、
training.bid_bank.derive_bid_bank_observation_normalization、EVEnv._preroll_initial_evs を差し替える。
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import pickle
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import realday as rd  # noqa: E402
import build_real  # noqa: E402

ROOT = rd.ROOT
MAIN_RUN = ROOT / 'archive' / 'wr_cap77wd_aemoplan_6station_resbid_tanh_ls717_20261012_102306'
LOCAL_STATE = HERE / 'main_state' / 'main_aemo6_training_state_ep2000.pth'
PATTERNS = rd.ALL_PATTERNS
WORLDS = PATTERNS + tuple(f'{k}_{p}' for k in ('fc', 'ml') for p in PATTERNS)
MODES = WORLDS + ('control', 'multi') + tuple(f'{k}_{w}' for k in ('scratch', 'multift') for w in WORLDS)


def main_state_path() -> Path:
    return LOCAL_STATE if LOCAL_STATE.exists() else MAIN_RUN / 'resume' / 'training_state_ep2000.pth'


def multi_state_path() -> Path:
    found = sorted((ROOT / 'archive').glob(f'{model_name("multi")}_2*'))
    if not found:
        raise FileNotFoundError('multi の実行がない')
    return found[-1] / 'resume' / 'training_state_ep2000.pth'


def model_name(mode: str) -> str:
    return f'realday_aemo6_{mode}'


def pattern_of(mode: str):
    """その日の世界で学習する mode なら、束と前の2日の名前（busy、ml_busy など）。scratch_ と multift_ は外して読む。"""
    for prefix in ('scratch_', 'multift_'):
        if mode.startswith(prefix):
            mode = mode[len(prefix):]
    return mode if mode in WORLDS else None


def start_state(mode: str):
    if mode in WORLDS or mode == 'control':
        return main_state_path()
    if mode.startswith('multift_'):
        return multi_state_path()
    return None


def configure(mode: str):
    sys.path.insert(0, str(rd.world.MAIN_STAGE))
    import stage_worker
    sys.modules['Config'] = types.ModuleType('Config')
    stage_worker.configure('aemo')
    del sys.modules['Config']
    pattern = pattern_of(mode)
    if mode == 'control':
        train, test = (rd.world.main_bank('aemo', s) for s in ('train', 'validation'))
    elif mode == 'multi':
        import build_multi
        train, test = (rd.world.BANKS / build_multi.bank_name(s) for s in ('train', 'validation'))
    else:
        train, test = (rd.world.BANKS / build_real.bank_name(pattern, s) for s in ('train', 'validation'))
    os.environ['EVMA_LOWER_TRAIN_BID_BANK_DIR'] = str(train)
    os.environ['EVMA_LOWER_TRAIN_TEST_BID_BANK_DIR'] = str(test)
    if mode.startswith('scratch_'):
        os.environ['EVMA_EPSILON_END_EPISODE'] = '500'
        os.environ['EVMA_OU_NOISE_END_EPISODE'] = '750'
    import Config
    stage_worker.check(Config, 'aemo')
    return train, test


def install(mode: str):
    import torch
    from training import bid_bank
    import training.train as T
    profile = json.loads((MAIN_RUN / 'input' / 'observation_normalization.json').read_text(encoding='utf-8'))
    bid_bank.derive_bid_bank_observation_normalization = lambda *a, **k: dict(profile)
    path = start_state(mode)
    if path is not None:
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if int(saved['completed_training_episode']) != 2000:
            raise RuntimeError(f'出発点の保存した状態が2000回目でない: {saved["completed_training_episode"]}')
        orig_build = T.build_marl_agent

        def build(*args, **kwargs):
            agent = orig_build(*args, **kwargs)
            info = agent.load_training_resume_state_dict(saved['agent'])
            base_ep = int(info['current_episode'])
            orig_start = agent.episode_start

            def episode_start():
                if not agent.test_mode and agent.current_episode < base_ep:
                    agent.current_episode += base_ep
                orig_start()

            agent.episode_start = episode_start
            print(f'[realft] loaded {path} (episode {saved["completed_training_episode"]}, '
                  f'replay {info["replay_size"]}, update_step {info["update_step"]})', flush=True)
            return agent

        T.build_marl_agent = build
    pattern = pattern_of(mode)
    if mode == 'multi':
        rd.install_preroll(None, by_probs=pickle.loads((HERE / 'prev_multi.pkl').read_bytes()))
    elif pattern is not None:
        rd.install_preroll(pickle.loads((HERE / f'prev_{pattern}.pkl').read_bytes()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=MODES)
    ap.add_argument('stage', choices=('preflight', 'pretrain'))
    ap.add_argument('--episodes', type=int, default=1000)
    args, extra = ap.parse_known_args()
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    train, test = configure(args.mode)
    import torch
    torch.set_num_threads(1)
    if args.stage == 'preflight':
        import Config
        state = None
        try:
            state = start_state(args.mode)
        except FileNotFoundError as e:
            state = f'missing: {e}'
        print(json.dumps({'mode': args.mode, 'model': model_name(args.mode), 'train_bank': train.name, 'test_bank': test.name,
                          'train_bank_exists': train.exists(), 'start_state': str(state),
                          'epsilon_end': Config.EPSILON_END_EPISODE, 'ou_end': Config.OU_NOISE_END_EPISODE,
                          'cuda': torch.cuda.is_available()}, ensure_ascii=False), flush=True)
        return 0
    install(args.mode)
    assert torch.cuda.is_available(), 'training must run on the GPU'
    module = importlib.import_module('pre_train')
    sys.argv = ['pre_train.py', '--episodes', str(args.episodes), '--model-name', model_name(args.mode),
                '--resume-checkpoint-interval', '200', '--bank-dir', str(train), '--test-bank-dir', str(test)] + extra
    return int(module.main() or 0)


if __name__ == '__main__':
    raise SystemExit(main())
