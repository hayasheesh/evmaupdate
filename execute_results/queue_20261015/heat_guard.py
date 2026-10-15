"""Windows の順番待ちの仕事と選び方の表を、CPU が熱いあいだ一時停止し、冷えたら再開する（2026-10-15〜）。

    python heat_guard.py          start_heat_guard.ps1 から WMI で起動する

温度は paper_eval_20261014/temperature_watch.py が30秒ごとに書く temperature_status.json を読む。
CPU 90 °C 以上か GPU 70 °C 以上（hot_reason）、または温度が90秒以上取れないと、対象のプロセスを一時停止する。
直近2分の CPU が80 °C 未満かつ GPU 60 °C 未満（cool_for_restart）になったら再開する。止める・再開する基準は
continue_work.py と同じ。continue_work.py が止めるのは以前の評価の仕事だけで、ここで扱う仕事は入っていない。
止めずに一時停止にするのは、入札の LP が1回で数十分〜数時間かかり、止めると途中までの計算が失われるため。

対象は、queue_20261015/queue.py と selection_table_6station_20261015/run.py を起動した cmd.exe の子孫の python.exe。
queue.py と run.py 自体も入るので、熱いあいだは次の仕事も始まらない。学習（GPU）は入れない。
この見張りが終わるときと、起動したときに、前に一時停止したプロセスを再開する（止めたまま残さない）。
記録は heat_guard_state.json（今の状態）と heat_guard_events.jsonl（一時停止と再開）。
"""
from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import time

import psutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TEMPERATURE = ROOT / 'execute_results' / 'paper_eval_20261014' / 'temperature_status.json'
STATE = HERE / 'heat_guard_state.json'
EVENTS = HERE / 'heat_guard_events.jsonl'
PARENT_MARKS = (('queue_20261015', 'queue.py'), ('selection_table_6station_20261015', 'run.py'))
INTERVAL_S = 15


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec='seconds')


def event(kind: str, **fields) -> None:
    with EVENTS.open('a', encoding='utf-8') as f:
        f.write(json.dumps({'time': now(), 'event': kind, **fields}, ensure_ascii=False) + '\n')


def temperature() -> tuple[str, dict]:
    """'hot', 'cool' or 'hold' (between the two thresholds), and the status read."""

    try:
        status = json.loads(TEMPERATURE.read_text(encoding='utf-8'))
        age = time.time() - TEMPERATURE.stat().st_mtime
    except (OSError, ValueError):
        return 'hot', {'error': 'temperature status unreadable'}
    if not status.get('valid') or age >= 90:
        return 'hot', {**status, 'age_s': round(age)}
    if status.get('hot_reason'):
        return 'hot', status
    if status.get('cool_for_restart'):
        return 'cool', status
    return 'hold', status


def targets() -> list[psutil.Process]:
    found = {}
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmd = ' '.join(proc.info['cmdline'] or [])
        except (psutil.Error, TypeError):
            continue
        if any(a in cmd and b in cmd for a, b in PARENT_MARKS):
            try:
                for child in proc.children(recursive=True):
                    child_cmd = ' '.join(child.cmdline())
                    if child.name().lower() != 'python.exe' or 'pretrain' in child_cmd:
                        continue
                    found[child.pid] = child
            except psutil.Error:
                continue
    return list(found.values())


def resume_all(pids) -> list[int]:
    resumed = []
    for pid in pids:
        try:
            psutil.Process(pid).resume()
            resumed.append(pid)
        except psutil.Error:
            pass
    return resumed


def write_state(paused: set[int], mode: str, status: dict) -> None:
    STATE.write_text(json.dumps({'time': now(), 'paused_pids': sorted(paused), 'mode': mode,
                                 'cpu_package_c': status.get('cpu_package_c'), 'gpu_c': status.get('gpu_c'),
                                 'hot_reason': status.get('hot_reason'), 'pid': os.getpid()},
                                ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    paused: set[int] = set()
    try:
        old = json.loads(STATE.read_text(encoding='utf-8')).get('paused_pids', [])
    except (OSError, ValueError):
        old = []
    if old:
        event('resume_left_over', pids=resume_all(old))
    event('start', pid=os.getpid())
    try:
        while True:
            mode, status = temperature()
            if mode == 'hot':
                new = []
                for proc in targets():
                    if proc.pid in paused:
                        continue
                    try:
                        proc.suspend()
                        paused.add(proc.pid)
                        new.append(proc.pid)
                    except psutil.Error:
                        pass
                if new:
                    event('pause', pids=new, reason=status.get('hot_reason') or status.get('error') or 'sensor',
                          cpu_c=status.get('cpu_package_c'), gpu_c=status.get('gpu_c'))
            elif mode == 'cool' and paused:
                event('resume', pids=resume_all(paused), cpu_c=status.get('cpu_package_c'), gpu_c=status.get('gpu_c'))
                paused.clear()
            write_state(paused, mode, status)
            time.sleep(INTERVAL_S)
    finally:
        if paused:
            event('resume_on_exit', pids=resume_all(paused))
            paused.clear()
            write_state(paused, 'stopped', {})


if __name__ == '__main__':
    main()
