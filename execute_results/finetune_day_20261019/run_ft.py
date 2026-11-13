"""finetune の学習1本を子プロセスで回し、状態を <mode>_run_status.json に書く（Windows）。

    python run_ft.py <day|control> [finetune_worker.py pretrain に渡す追加の引数 ...]

記録は logs/<mode>_pretrain.{stdout,stderr}.log。
"""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
CREATE_NO_WINDOW = 0x08000000


def now() -> str:
    return datetime.now().isoformat(timespec='seconds')


def main() -> int:
    mode, extra = sys.argv[1], sys.argv[2:]
    status_path = RUN / f'{mode}_run_status.json'
    state = {'mode': mode, 'started_at': now(), 'args': extra}

    def save():
        status_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    out = (RUN / 'logs' / f'{mode}_pretrain.stdout.log').open('a', encoding='utf-8')
    err = (RUN / 'logs' / f'{mode}_pretrain.stderr.log').open('a', encoding='utf-8')
    proc = subprocess.Popen([sys.executable, '-X', 'utf8', '-u', str(RUN / 'finetune_worker.py'), mode, 'pretrain', *extra],
                            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err, creationflags=CREATE_NO_WINDOW)
    state.update(state='running', pid=proc.pid)
    save()
    code = proc.wait()
    state.update(state='done', exit_code=code, finished_at=now(), result='complete' if code == 0 else 'failed')
    save()
    return code


if __name__ == '__main__':
    raise SystemExit(main())
