"""最初の数エピソードの値を1回ずつ並べる（読むだけ）。usage: early.py RUN_DIR [N]"""
import sys
from health import scalars, TAGS
run = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 12
d = scalars(run)
keys = ['Qg', 'gfrac', 'cos', 'gc_clip', 'gc_raw', 'gc_loss', 'lc_loss', 'td']
print('ep  ' + ' '.join(f'{k:>8}' for k in keys))
for ep in range(n):
    vals = [d.get(TAGS[k], {}).get(ep, float('nan')) for k in keys]
    if all(v != v for v in vals):
        continue
    print(f'{ep:<3} ' + ' '.join(f'{v:8.3f}' for v in vals))
