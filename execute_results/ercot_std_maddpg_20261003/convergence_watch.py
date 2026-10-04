"""ERCOT の標準 MADDPG の学習を見張り、収束したら止める。読むのと、止める依頼を書くことしかしない。

10分ごとに途中テストの記録（results/test_history.json）と TensorBoard の記録を読む。
  収束の判定  1000回以上学習し、かつ直近400回（途中テスト20回）のテストの和（追従率 + SoC 達成率）の
              傾きの絶対値が 100回あたり 1.0 以下になったとき。完全な状態を保存して止める依頼（STOP_REQUESTED）を書く。
  発散の兆し  批評家の Q（Q/local_mean）や損失（Loss/local_critic_mean）が非有限、|Q| > 100、損失 > 20 のとき。
              止めずに alarm を書き残すだけ（設定の調整は人が判断する）。
2000回に届いたら学習は自分で終わるので、見張りも終わる。
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SLOPE_LIMIT = 1.0          # 100回あたり
MIN_EPISODES = 1000
WINDOW = 400
INTERVAL_S = 600
STATE = HERE / 'convergence_watch.json'
LOG = HERE / 'convergence_watch.log'

spec = importlib.util.spec_from_file_location('health', ROOT / 'execute_results/watch_20261002/health.py')
health = importlib.util.module_from_spec(spec)
spec.loader.exec_module(health)
sys.path.insert(0, str(HERE / 'runtime_source'))
from training.training_resume import request_stop  # noqa: E402


def log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def save(state: dict) -> None:
    tmp = STATE.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(STATE)


def last_scalars(run: Path) -> dict:
    out = {}
    try:
        values = health.scalars(str(run))
        for key, tag in (('q', 'Q/local_mean'), ('loss', 'Loss/local_critic_mean')):
            series = values.get(tag) or {}
            if series:
                ep = max(series)
                out[key] = (ep, float(series[ep]))
    except Exception as exc:  # 記録の読み込みに失敗しても見張りは続ける
        out['error'] = repr(exc)
    return out


def main() -> None:
    state = {'started_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'slope_limit_per_100': SLOPE_LIMIT,
             'min_episodes': MIN_EPISODES, 'window': WINDOW, 'stage': 'watching', 'alarms': []}
    save(state)
    while True:
        status = json.loads((HERE / 'status.json').read_text(encoding='utf-8-sig'))
        if status.get('stage') != 'training':
            state['stage'] = f"learner_{status.get('stage')}"
            save(state)
            log(f"learner stage {status.get('stage')}; watcher exits")
            return
        run = Path(json.loads((HERE / 'run_dir.json').read_text(encoding='utf-8'))['runDir'])
        tests = health.tests(str(run))
        sc = last_scalars(run)
        state['last_check'] = time.strftime('%Y-%m-%dT%H:%M:%S')
        state['last_scalars'] = sc
        for key, limit in (('q', 100.0), ('loss', 20.0)):
            if key in sc:
                ep, value = sc[key]
                if not math.isfinite(value) or abs(value) > limit:
                    alarm = f'{key} {value} at ep {ep}'
                    if alarm not in state['alarms']:
                        state['alarms'].append(alarm)
                        log(f'[alarm] {alarm}')
        if tests:
            last_ep = max(tests)
            recent = sorted(ep for ep in tests if ep > last_ep - WINDOW)
            sums = [tests[ep][0] + tests[ep][1] for ep in recent]
            state['last_test_episode'] = last_ep
            state['last_test_sum'] = sums[-1]
            if len(recent) >= 2:
                slope = float(np.polyfit(np.asarray(recent, float), np.asarray(sums, float), 1)[0] * 100.0)
                state['slope_per_100'] = slope
                state['tests_in_window'] = len(recent)
                if last_ep >= MIN_EPISODES and len(recent) >= WINDOW // 20 and abs(slope) <= SLOPE_LIMIT:
                    path = request_stop(run)
                    state['stage'] = 'stop_requested'
                    state['stop_requested_at_episode'] = last_ep
                    state['stop_reason'] = f'|slope| {abs(slope):.2f} <= {SLOPE_LIMIT} per 100 episodes over the last {WINDOW}'
                    state['recent_sums'] = dict(zip(map(str, recent), sums))
                    save(state)
                    log(f'converged at test ep {last_ep}: slope {slope:.2f}/100, sum {sums[-1]:.1f}; wrote {path}')
                    return
        save(state)
        time.sleep(INTERVAL_S)


if __name__ == '__main__':
    main()
