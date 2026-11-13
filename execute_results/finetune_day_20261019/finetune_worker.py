"""本線の行動器（AEMO 6 station、2000回）から、その日の世界で学習を続ける（finetune、Windows、2026-10-19）。

    python finetune_worker.py <day|count|control> <preflight|pretrain> [--episodes 1000]

  day      その日の世界（dayworld.day_world、kappa と tau は build_day.CHOSEN）と、その日の入札だけの束
           （build_day.fake_bank_name）で学習を続ける。中間テストもその日の世界とその日の入札。
  count    その日の到着台数を中心にした世界（countworld.count_world、k は build_count.K）と、その世界で作った入札だけの束
           （build_count.count_bank_name）で学習を続ける。来たEVの中身は本線と同じく毎回引き直す。
  control  同じ回数を、本線の世界と本線の束で続ける（学習を足しただけの効果と分けるため）。
どちらも、学習の始めに本線の重み（main_aemo6_ep2000_networks.pt：行動器・批評家・目標のネットワークと最適化の状態）を読み込み、
探索の大きさ（epsilon、OU のノイズ）と回数の数えは本線の2000回目から続ける（2001回目以降は epsilon 0.005、OU 0.20）。
リプレイは空から貯め直す（最初の7000ステップは学習せずに貯める。行動は読み込んだ行動器にノイズを足したもの）。
観測の正規化は、本線の実行の input/observation_normalization.json を使う（束から作り直さない）。
共有のコードは変えず、このプロセスの中だけで training.train.build_marl_agent と
training.bid_bank.derive_bid_bank_observation_normalization と EVEnv の到着の作り方を差し替える。
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dayworld as dw  # noqa: E402
import build_day  # noqa: E402

ROOT = dw.ROOT
MAIN_RUN = ROOT / 'archive' / 'wr_cap77wd_aemoplan_6station_resbid_tanh_ls717_20261012_102306'
NETWORKS = HERE / 'main_aemo6_ep2000_networks.pt'


def configure(mode: str):
    sys.path.insert(0, str(dw.world.MAIN_STAGE))
    import stage_worker
    sys.modules['Config'] = types.ModuleType('Config')
    stage_worker.configure('aemo')
    del sys.modules['Config']
    if mode == 'day':
        train, test = (dw.world.BANKS / build_day.fake_bank_name(s) for s in ('train', 'validation'))
    elif mode == 'count':
        import build_count
        train, test = (dw.world.BANKS / build_count.count_bank_name(s) for s in ('train', 'validation'))
    else:
        train, test = (dw.world.main_bank('aemo', s) for s in ('train', 'validation'))
    os.environ['EVMA_LOWER_TRAIN_BID_BANK_DIR'] = str(train)
    os.environ['EVMA_LOWER_TRAIN_TEST_BID_BANK_DIR'] = str(test)
    import Config
    stage_worker.check(Config, 'aemo')
    return train, test


def install(mode: str):
    import torch
    from training import bid_bank
    import training.train as T
    profile = json.loads((MAIN_RUN / 'input' / 'observation_normalization.json').read_text(encoding='utf-8'))
    bid_bank.derive_bid_bank_observation_normalization = lambda *a, **k: dict(profile)
    saved = torch.load(NETWORKS, map_location='cpu', weights_only=False)
    orig_build = T.build_marl_agent

    def build(*args, **kwargs):
        agent = orig_build(*args, **kwargs)
        compat = agent._training_resume_compatibility()
        if saved['compatibility'] != compat:
            raise RuntimeError('本線の重みと今の行動器の作りが違う')
        models, optim, scalars = saved['models'], saved['optimizers'], saved['scalars']
        agent._load_module_list(agent.actors, models['actors'], 'actor')
        agent._load_module_list(agent.t_actors, models['target_actors'], 'target actor')
        agent._load_module_list(agent.critics, models['critics'], 'critic')
        agent._load_module_list(agent.critics2, models['critics2'], 'critic2')
        agent._load_module_list(agent.t_critics, models['target_critics'], 'target critic')
        agent._load_module_list(agent.t_critics2, models['target_critics2'], 'target critic2')
        agent.global_critic1.load_state_dict(models['global_critic1'])
        agent.t_global_critic1.load_state_dict(models['target_global_critic1'])
        agent.global_critic2.load_state_dict(models['global_critic2'])
        agent.t_global_critic2.load_state_dict(models['target_global_critic2'])
        agent._load_optimizer_list(agent.opt_a, optim['actors'], 'actor')
        agent._load_optimizer_list(agent.opt_c, optim['critics'], 'critic')
        agent._load_optimizer_list(agent.opt_c2, optim['critics2'], 'critic2')
        agent.opt_global_c1.load_state_dict(optim['global_critic1'])
        agent.opt_global_c2.load_state_dict(optim['global_critic2'])
        agent.current_episode = int(scalars['current_episode'])
        agent.epsilon = float(scalars['epsilon'])
        agent.ou_noise_scale = float(scalars['ou_noise_scale'])
        agent.update_step = int(scalars['update_step'])
        # 学習の始めに train.py が回数の数えを0に戻すので、エピソードの始めに本線の回数を足し直す（探索の大きさは2001回目からの値になる）
        base_ep = int(scalars['current_episode'])
        orig_start = agent.episode_start

        def episode_start():
            if not agent.test_mode and agent.current_episode < base_ep:
                agent.current_episode += base_ep
            orig_start()

        agent.episode_start = episode_start
        print(f'[finetune] loaded main networks from {saved["source"]} (episode {saved["completed_training_episode"]})', flush=True)
        return agent

    T.build_marl_agent = build
    if mode == 'day':
        base = dw.target_base()
        D = dw.center(base)
        dw.day_world(D, build_day.CHOSEN['kappa'], build_day.CHOSEN['tau']).__enter__()  # このプロセスのすべての1日をその日の世界にする
    elif mode == 'count':
        import build_count
        import countworld
        n_day = countworld.day_counts(dw.center(dw.target_base()))
        countworld.count_world(n_day, build_count.K).__enter__()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('day', 'count', 'control'))
    ap.add_argument('stage', choices=('preflight', 'pretrain'))
    ap.add_argument('--episodes', type=int, default=1000)
    args, extra = ap.parse_known_args()
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    train, test = configure(args.mode)
    import torch
    torch.set_num_threads(1)
    install(args.mode)
    model = f'ftday_aemo6_20240201_{args.mode}'
    if args.stage == 'preflight':
        print(json.dumps({'mode': args.mode, 'model': model, 'train_bank': train.name, 'test_bank': test.name,
                          'cuda': torch.cuda.is_available()}, ensure_ascii=False), flush=True)
        return 0
    assert torch.cuda.is_available(), 'training must run on the GPU'
    module = importlib.import_module('pre_train')
    sys.argv = ['pre_train.py', '--episodes', str(args.episodes), '--model-name', model,
                '--resume-checkpoint-interval', '100', '--bank-dir', str(train), '--test-bank-dir', str(test)] + extra
    return int(module.main() or 0)


if __name__ == '__main__':
    raise SystemExit(main())
