"""Windows の 6 station：蓄電池の大きさを振る評価（run_rule_bess.py）が終わるのを待ってから、
表3（lp・rule・marl_force・marl、各40組）と、図4の入札の変種（検証用5日）と未見通過率を回す。

    python run_windows.py      （start_windows.ps1 が WMI で画面を出さずに起動する）

記録は logs/ に、終わった印は DONE_table3 / DONE_<task>。
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
NO_WINDOW = 0x08000000


def bess_running() -> bool:
    out = subprocess.run(['C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe', '-NoProfile', '-Command',
                          "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match 'run_rule_bess' }).Count"],
                         capture_output=True, text=True, creationflags=NO_WINDOW).stdout.strip()
    return out not in ('', '0')


def run(args, name):
    logs = HERE / 'logs'
    logs.mkdir(exist_ok=True)
    with open(logs / f'{name}.log', 'a', encoding='utf-8') as out, open(logs / f'{name}.err', 'a', encoding='utf-8') as err:
        return subprocess.Popen([PY, '-X', 'utf8', '-u', *args], cwd=HERE, stdout=out, stderr=err,
                                stdin=subprocess.DEVNULL, creationflags=NO_WINDOW)


def main():
    while bess_running():
        time.sleep(120)
    table3 = [(t, m) for m in ('rule', 'marl_force', 'marl', 'lp') for t in ('aemo', 'ercot')]
    streams = {}
    for t in ('aemo', 'ercot'):
        base = f'wr_cap77wd_min70_margin7_validation_5_6station_128cmd_3ev_{t}_plan_deviation'
        streams[t] = [
            (['build_variant_bank.py', t, 'session_count', '128', '2', '3'], f'bank_{t}'),
            (['build_variant_bank.py', t, 'low_connection_2_max_count', '256', '2', '3'], f'bank_{t}'),
            (['run_unseen_variants.py', t, f'unseen_{t}', '6', base, base.replace('_128cmd_3ev_', '_128cmd_minmedmax_'),
              base.replace('_128cmd_3ev_', '_256cmd_low2max_')], f'unseen_{t}'),
        ]
    streams['table3'] = [(['run_table3.py', t, m, '4', '2', '8'], 'table3') for t, m in table3]
    current = {k: None for k in streams}
    while any(streams.values()) or any(p is not None for p in current.values()):
        for k in streams:
            p = current[k]
            if p is not None and p.poll() is None:
                continue
            if streams[k]:
                args, name = streams[k].pop(0)
                current[k] = run(args, name)
            else:
                if p is not None:
                    (HERE / f'DONE_{k}').touch()
                current[k] = None
        time.sleep(30)


if __name__ == '__main__':
    main()
