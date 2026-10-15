"""Windows で作り終えた入札の束を、研究室の bid_banks へ写す（Windows のキューから使う）。

    python copy_bank_to_lab.py [--portable] <bank名> [...]

--portable を付けると、make_portable_bank.py で研究室向けに書き直した execute_results/bid_banks_portable/<bank名> を写す。

束ごとに manifest.json があることを確かめてから、まず <bank名>.copying に写し、最後に名前を付け替える。
研究室の側は manifest.json の有無で学習を始めるので、途中まで写った束を読むことはない。
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
LAB_BANKS = '/home/hayashi/workspace/EVMALOCALUPDATE/execute_results/bid_banks'


def main():
    names = sys.argv[1:]
    source = 'bid_banks'
    if names and names[0] == '--portable':
        names, source = names[1:], 'bid_banks_portable'
    for name in names:
        local = ROOT / 'execute_results' / source / name
        if not (local / 'manifest.json').exists():
            raise SystemExit(f'作り終えていない: {name}')
        tmp = f'{LAB_BANKS}/{name}.copying'
        subprocess.run(['ssh', '-o', 'BatchMode=yes', 'smartgrid-gpu', f'rm -rf {tmp} && mkdir -p {tmp}'], check=True)
        subprocess.run(['scp', '-q', '-r', '-o', 'BatchMode=yes', *[str(p) for p in local.iterdir()],
                        f'smartgrid-gpu:{tmp}/'], check=True)
        subprocess.run(['ssh', '-o', 'BatchMode=yes', 'smartgrid-gpu',
                        f'test -f {tmp}/manifest.json && test ! -e {LAB_BANKS}/{name} && mv {tmp} {LAB_BANKS}/{name}'],
                       check=True)
        print('copied', name, flush=True)


if __name__ == '__main__':
    main()
