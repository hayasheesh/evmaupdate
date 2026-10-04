"""hobo100100 と任意の run を、報酬・Q・勾配の桁で並べる。

usage: python compare_to_hobo.py <run_dir> [max_ep]
"""
import glob, statistics, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAGS = [
    ('Reward/global', '報酬global'),
    ('Q/global', 'Q global'),
    ('Gradient/global_critic_raw', 'gcrit生'),
    ('Loss/global_critic', 'gcrit loss'),
    ('Clipping/global_critic', 'gcrit clip'),
    ('Gradient/local_critic_raw_agent1', 'lcrit生'),
    ('Loss/local_critic_mean', 'lcrit loss'),
    ('Q/local_mean', 'Q local'),
    ('Gradient/actor_raw_agent1', 'actor生'),
    ('Reward/local', '報酬local'),
]


def load(paths):
    out = {}
    for p in paths:
        ea = EventAccumulator(p, size_guidance={'scalars': 0})
        ea.Reload()
        for t in ea.Tags()['scalars']:
            for e in ea.Scalars(t):
                out.setdefault(t, {})[e.step] = e.value
    return out


def bucket(S, tag, lo, hi):
    v = [S[tag][s] for s in S.get(tag, {}) if lo <= s < hi]
    return statistics.mean(v) if v else None


def main():
    run_dir = sys.argv[1]
    max_ep = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    H = load(glob.glob("docs/hobo100100/performance/*"))
    R = load(glob.glob(f"{run_dir}/performance/*"))
    steps = sorted(R.get('Q/global', {}))
    print(f"run = {run_dir}")
    print(f"到達 ep = {steps[-1] if steps else 0}")
    print()
    for tag, label in TAGS:
        print(f"--- {label} ({tag})")
        print(f"{'ep':>6} {'hobo':>12} {'この run':>12} {'比':>9}")
        for b in range(0, max_ep + 1, 25):
            h = bucket(H, tag, b, b + 25)
            r = bucket(R, tag, b, b + 25)
            if r is None:
                continue
            ratio = (f"{r/h:>9.2f}" if h not in (None, 0) else f"{'-':>9}")
            hs = f"{h:>12.4f}" if h is not None else f"{'-':>12}"
            print(f"{b:>6} {hs} {r:>12.4f} {ratio}")
        print()


if __name__ == "__main__":
    main()
