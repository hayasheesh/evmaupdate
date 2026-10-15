"""1つの市場の変種の入札をすべて作り、未見通過率を測る。途中で止めても、作り終えた入札と測り終えた日は飛ばして続ける。

    python run.py <aemo|ercot> <同時に作る入札の数> <未見の判定の並列数>

入札1つは scenario-workers 3 で作るので、使うコアは「同時に作る入札の数 × 3」。
ログは logs/ に、子プロセスの出力の各行に時刻を付けて書く（入札の1日ごとの時間は report で行の時刻から求める）。
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import table  # noqa: E402

BUILD_ORDER = ['cmd8', 'cmd16', 'cmd32', 'cmd64', 'cnt1', 'cnt2', 'sel_low1medmax', 'sel_random0', 'sel_random1',
               'sel_random2', 'cnt4', 'cnt6', 'acc95_low2max', 'acc91_low2max', 'acc85_low2max',
               'acc95_random0', 'acc91_random0', 'acc85_random0']
UNSEEN_ORDER = ['main', 'cmd256', 'sel_minmedmax'] + BUILD_ORDER


def status(task, **kw):
    p = HERE / f'status_{task}.json'
    p.write_text(json.dumps(dict(time=datetime.now().astimezone().isoformat(timespec='seconds'), **kw),
                            ensure_ascii=False, indent=2), encoding='utf-8')


def run_logged(args, log):
    log.parent.mkdir(exist_ok=True)
    with log.open('a', encoding='utf-8') as f:
        f.write(f'{datetime.now().isoformat(timespec="seconds")} START {" ".join(args)}\n')
        p = subprocess.Popen([sys.executable, '-X', 'utf8', '-u', *args], cwd=HERE, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
        for line in p.stdout:
            f.write(f'{datetime.now().isoformat(timespec="seconds")} {line}')
            f.flush()
        code = p.wait()
        f.write(f'{datetime.now().isoformat(timespec="seconds")} EXIT {code}\n')
    return code


def build_jobs(task):
    jobs = []
    for variant in BUILD_ORDER:
        seeds = [None] if table.VARIANTS[variant][2] is None else list(range(table.EVAL_SEEDS))
        for j in seeds:
            out = table.BANKS / table.bank_name(task, variant, j)
            if (out / 'BUILD_SECONDS').exists():
                continue
            args = ['table.py', 'build', task, variant] + ([] if j is None else [str(j)])
            jobs.append((variant, j, args, HERE / 'logs' / f'build_{task}_{variant}{"" if j is None else f"_s{j}"}.log'))
    return jobs


def unseen_done(task, variant):
    p = HERE / f'unseen_{task}_{variant}.jsonl'
    if not p.exists():
        return False
    rows = {(r['bank'], r['day'], r.get('chunk')) for r in map(json.loads, p.read_text(encoding='utf-8').splitlines())}
    return len(rows) == 5 * (128 // table.CHUNK) * len(table.bank_names(task, variant))


def main():
    task, concurrency, workers = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    jobs = build_jobs(task)
    status(task, stage='build', remaining=len(jobs))
    failed = []
    with ThreadPoolExecutor(concurrency) as pool:
        futures = {pool.submit(run_logged, args, log): (v, j) for v, j, args, log in jobs}
        for fut, key in futures.items():
            if fut.result() != 0:
                failed.append(key)
    if failed:
        status(task, stage='build_failed', failed=[f'{v} s{j}' for v, j in failed])
        raise SystemExit(f'入札を作れなかった: {failed}')
    for variant in UNSEEN_ORDER:
        if unseen_done(task, variant):
            continue
        if not all((table.BANKS / name).exists() for name in table.bank_names(task, variant)):
            print('この計算機に入札がないので飛ばす:', task, variant, flush=True)
            continue
        status(task, stage='unseen', variant=variant)
        if run_logged(['table.py', 'unseen', task, variant, str(workers)], HERE / 'logs' / f'unseen_{task}_{variant}.log') != 0:
            status(task, stage='unseen_failed', variant=variant)
            raise SystemExit(f'未見の判定が止まった: {variant}')
    status(task, stage='done')
    (HERE / f'DONE_{task}').write_text(datetime.now().isoformat(timespec='seconds') + '\n')


if __name__ == '__main__':
    main()
