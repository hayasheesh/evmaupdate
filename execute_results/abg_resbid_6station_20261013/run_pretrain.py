"""1つの task の学習（pretrain）だけを子プロセスで回し、状態を <task>_run_status.json に書く（Windows）。

    python run_pretrain.py <task> [pre_train に渡す追加の引数 ...]

入札は Windows にある本線の入札（resbid_6station_20261012 と同じ）を使い、入札の段（train_bank / test_bank）は起動しない。
入札の段を起動すると、入札がそろっていても目録の更新時刻が書き換わり、同じ入札を使うほかの実行（通常 MADDPG など）が
保存した状態から再開できなくなるため。記録は logs/<task>_pretrain.{stdout,stderr}.log。
状態の書き方は run_task.py と同じ（見張り watch_quiet/watch.sh が result を読む）。
"""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
WORKER = RUN / 'stage_worker.py'
CREATE_NO_WINDOW = 0x08000000


def now() -> str:
    return datetime.now().isoformat(timespec='seconds')


def main() -> int:
    task, extra = sys.argv[1], sys.argv[2:]
    status_path = RUN / f'{task}_run_status.json'
    state = {'task': task, 'started_at': now(), 'stages': {}}

    def save():
        status_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    out = (RUN / 'logs' / f'{task}_pretrain.stdout.log').open('a', encoding='utf-8')
    err = (RUN / 'logs' / f'{task}_pretrain.stderr.log').open('a', encoding='utf-8')
    proc = subprocess.Popen([sys.executable, '-X', 'utf8', '-u', str(WORKER), task, 'pretrain', *extra],
                            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err, creationflags=CREATE_NO_WINDOW)
    state['stages']['pretrain'] = {'state': 'running', 'pid': proc.pid, 'started_at': now()}
    save()
    code = proc.wait()
    state['stages']['pretrain'].update(state='done' if code == 0 else 'failed', exit_code=code, finished_at=now())
    state['result'] = 'complete' if code == 0 else 'pretrain failed'
    save()
    return code


if __name__ == '__main__':
    raise SystemExit(main())
