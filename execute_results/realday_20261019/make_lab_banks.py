"""研究室で、build_real.py が作ったその日の入札（bid_<pattern>.pkl）から finetune 用の束を作る（目録は研究室の本線の束から写す）。

    python make_lab_banks.py busy quiet
"""
from __future__ import annotations

import json
import pickle
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import realday as rd  # noqa: E402
import build_real  # noqa: E402


def main():
    rd.world.configure(rd.TASK)
    from training.bid_bank import save_fixed_bid
    for pattern in sys.argv[1:]:
        bid = pickle.loads((HERE / f'bid_{pattern}.pkl').read_bytes())
        for split in ('train', 'validation'):
            src = rd.world.main_bank(rd.TASK, split)
            dst = rd.world.BANKS / build_real.bank_name(pattern, split)
            if dst.exists():
                raise FileExistsError(dst)
            dst.mkdir(parents=True)
            shutil.copy2(src / 'manifest.json', dst / 'manifest.json')
            manifest = json.loads((src / 'manifest.json').read_text(encoding='utf-8'))
            for entry in manifest['entries']:
                save_fixed_bid(dst / Path(str(entry['bid_path']).replace(chr(92), '/')), bid)
            (dst / 'REALDAY.json').write_text(json.dumps({'pattern': pattern, 'days': bid['realday']['days']}, ensure_ascii=False),
                                              encoding='utf-8')
            print('bank', dst.name, len(manifest['entries']), 'entries', flush=True)


if __name__ == '__main__':
    main()
