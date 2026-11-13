"""元データの実際の1日を使う finetune の部品（AEMO 6 station、Windows と研究室、2026-10-19）。

6 station の並びは JPL、OSL、Caltech、TRO、JPL、OSL（EVMA_STATION_ROSTER=workplace_residential）。4つのデータ源は記録期間が
重ならないので、その日はデータ源ごとに別の日付を選んで組む（利用者の決定）。

日の選び方（PATTERNS）：データ源ごとに、平日の1日あたりのセッション数で並べ、混んだ日は上から10%の順位、空いた日は下から10%、
平均的な日は中央の順位に一番近い日。同じデータ源の2局（JPL の局0と局4、OSL の局1と局5）には、その順位の近くの別の日を当てる。
前の2日がそのデータ源の記録期間の中にある日だけを選ぶ。

その日の量（前日に分かるもの）：局ごと・1時間ごとのその日の実際の件数 n_h から、本線と同じ式で到着の確率を作る
（1つの口の5分ごとの確率 = n_h ÷ 口数 × 増加率 ÷ 12。本線は n_h の代わりに平日の平均を使う）。前の2日も同じ式で作る。
  学習の世界：その日の確率で本線と同じくじ引きをし（入札に確率を入れれば、学習の reset はそれを使う）、0時にいるEVは、
  前の2日の確率で本線と同じ流し方をする（install_preroll）。来た車の中身は本線と同じ一般の分布から毎回引く。
本当の状況（本番のテストだけで使う）：その日と前の2日の実際のセッションを、本線と同じ倍率（10口 × 増加率 ÷ 実際の口数）で
  引き直して並べる（種は1つに固定）。来た時刻、滞在時間、電力量は実際のもの、SoC と電池の容量と定格は本線と同じ分布から引く。
  0時にいるEVは、前の2日の実際のセッションを充電口の数の制限つきで流して、0時に残っているもの。
"""
from __future__ import annotations

import copy
import json
import math
import random as _r
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FCW = ROOT / 'execute_results' / 'fcworld_6station_20261018'
sys.path.insert(0, str(FCW))
import world  # noqa: E402

TASK = 'aemo'
DAY_INDEX = 1                      # 入札の指令と種は、本線の検証用の束の2日目（2024-02-01）のものを使う
ROSTER = ('ACN_JPL_ARROYO1', 'RESIDENTIAL NORWAY _ OSL_S', 'ACN_CALTECH_GARAGE1', 'RESIDENTIAL NORWAY _ TRO_R',
          'ACN_JPL_ARROYO1', 'RESIDENTIAL NORWAY _ OSL_S')
PATTERNS = {'busy': 0.90, 'quiet': 0.10, 'average': 0.50}
# 足したテストの日（choose_more_days、days_more.json）。名前の数字は、平日の件数の順位の下からの割合（%）。
MORE_PATTERNS = {'q05': 0.05, 'q20': 0.20, 'q35': 0.35, 'q65': 0.65, 'q80': 0.80, 'q95': 0.95}
ALL_PATTERNS = tuple(PATTERNS) + tuple(MORE_PATTERNS)
PREV_DAYS = 2
SESSION_DIR = ROOT / 'data' / 'input_EVinfo' / 'station_sessions'
REAL_SEED = 20261019


def target_base():
    from training.bid_bank import BidBank
    bank = BidBank(world.main_bank(TASK, 'validation'))
    return bank.load_entry(bank.entries[DAY_INDEX])


def _frames():
    import pandas as pd
    meta = json.loads((SESSION_DIR / 'metadata.json').read_text(encoding='utf-8'))['stations']
    out = {}
    for sid in sorted(set(ROSTER)):
        info = meta[sid]
        frame = pd.read_csv(SESSION_DIR / info['file'])
        out[sid] = (info, frame)
    return out


def choose_days():
    """パターンごとに、局ごとの日付（6局）と、その日の件数・順位。"""
    import datetime as dt
    import numpy as np
    frames = _frames()
    out = {}
    for name, q in PATTERNS.items():
        used = {}
        days = []
        for st, sid in enumerate(ROSTER):
            info, frame = frames[sid]
            dates = sorted(frame['local_date'].unique())
            first, last = dt.date.fromisoformat(dates[0]), dt.date.fromisoformat(dates[-1])
            wk = frame[frame['day_class'] == 'weekday'].groupby('local_date').size()
            ok = [d for d in wk.index if dt.date.fromisoformat(d) - dt.timedelta(days=PREV_DAYS) >= first]
            wk = wk.loc[ok].sort_values(kind='mergesort')
            rank = q * (len(wk) - 1)
            order = sorted(range(len(wk)), key=lambda i: (abs(i - rank), i))
            taken = used.setdefault(sid, set())
            i = next(i for i in order if wk.index[i] not in taken)
            taken.add(wk.index[i])
            days.append({'station': st, 'source': sid, 'date': wk.index[i], 'sessions': int(wk.iloc[i]),
                         'rank': i, 'weekdays': len(wk), 'median_sessions': float(np.median(wk.values)),
                         'ports': int(info['ports'])})
        out[name] = days
    return out


def load_days(name):
    """パターンの局ごとの日付（days.json、なければ days_more.json）。"""
    for fn in ('days.json', 'days_more.json'):
        p = HERE / fn
        if p.exists():
            days = json.loads(p.read_text(encoding='utf-8'))
            if name in days:
                return days[name]
    raise KeyError(name)


def _forecastable(frame, date):
    """その日と前日に、前日の昼の予測（forecast_counts）が作れるか。"""
    p1 = _prev_dates(date)[-1]
    try:
        forecast_counts(frame, date, _class_of(frame, date))
        forecast_counts(frame, p1, _class_of(frame, p1))
    except ValueError:
        return False
    return True


def choose_more_days():
    """足すテストの日（MORE_PATTERNS）。選び方は choose_days と同じ（データ源ごとに平日の件数の順位で、その分位点に一番近い日）。
    ただし、いろいろな日の学習（multi_days.json、E）と days.json の日付、前日の昼の予測が作れない日は選ばない。
    E がその日を見ていないようにするため。足す日どうしでも、同じデータ源の日付は重ねない。"""
    import datetime as dt
    import numpy as np
    frames = _frames()
    test = json.loads((HERE / 'days.json').read_text(encoding='utf-8'))
    multi = json.loads((HERE / 'multi_days.json').read_text(encoding='utf-8'))
    banned = ({(d['source'], d['date']) for rows in test.values() for d in rows}
              | {(d['source'], d['date']) for day in multi for d in day})
    taken = {}
    out = {}
    for name, q in MORE_PATTERNS.items():
        days = []
        for st, sid in enumerate(ROSTER):
            info, frame = frames[sid]
            dates = sorted(frame['local_date'].unique())
            first = dt.date.fromisoformat(dates[0])
            wk = frame[frame['day_class'] == 'weekday'].groupby('local_date').size()
            ok = [d for d in wk.index if dt.date.fromisoformat(d) - dt.timedelta(days=PREV_DAYS) >= first]
            wk = wk.loc[ok].sort_values(kind='mergesort')
            rank = q * (len(wk) - 1)
            order = sorted(range(len(wk)), key=lambda i: (abs(i - rank), i))
            used = taken.setdefault(sid, set())
            i = next(i for i in order
                     if wk.index[i] not in used and (sid, wk.index[i]) not in banned and _forecastable(frame, wk.index[i]))
            used.add(wk.index[i])
            days.append({'station': st, 'source': sid, 'date': wk.index[i], 'sessions': int(wk.iloc[i]),
                         'rank': i, 'target_rank': float(rank), 'weekdays': len(wk),
                         'median_sessions': float(np.median(wk.values)), 'ports': int(info['ports'])})
        out[name] = days
    return out


def choose_multi_days(n_days=30, seed=REAL_SEED):
    """いろいろな日で学習するための組んだ日（n_days 日、局ごとに平日からランダムに選ぶ）。本番のテストの日付（days.json の
    3パターンの全日付）と、前の2日が記録期間の外になる日は除く。同じ組んだ日の中で、同じデータ源の2局は別の日にする。"""
    import datetime as dt
    rng = _r.Random(seed)
    frames = _frames()
    test = json.loads((HERE / 'days.json').read_text(encoding='utf-8'))
    banned = {(d['source'], d['date']) for rows in test.values() for d in rows}
    pools = {}
    for sid, (info, frame) in frames.items():
        dates = sorted(frame['local_date'].unique())
        first = dt.date.fromisoformat(dates[0])
        wk = frame[frame['day_class'] == 'weekday'].groupby('local_date').size()
        pools[sid] = [(d, int(n)) for d, n in wk.items()
                      if (sid, d) not in banned and dt.date.fromisoformat(d) - dt.timedelta(days=PREV_DAYS) >= first]
    out = []
    for k in range(n_days):
        used = {}
        day = []
        for st, sid in enumerate(ROSTER):
            info, _frame = frames[sid]
            taken = used.setdefault(sid, set())
            date, n = rng.choice([x for x in pools[sid] if x[0] not in taken])
            taken.add(date)
            day.append({'station': st, 'source': sid, 'date': date, 'sessions': n, 'ports': int(info['ports'])})
        out.append(day)
    return out


def _date_rows(frame, date):
    return frame[frame['local_date'] == date].sort_values('arrival_minute', kind='mergesort')


def _class_of(frame, date):
    import datetime as dt
    rows = frame[frame['local_date'] == date]
    if len(rows):
        return str(rows['day_class'].iloc[0])
    return 'holiday' if dt.date.fromisoformat(date).weekday() >= 5 else 'weekday'


def _prev_dates(date):
    import datetime as dt
    d = dt.date.fromisoformat(date)
    return [str(d - dt.timedelta(days=k)) for k in range(PREV_DAYS, 0, -1)]


def hourly_probs(frame, date, ports, growth, steps=288):
    """その日の実際の件数から作る、1つの口の5分ごとの到着の確率（本線と同じ式）。"""
    import numpy as np
    rows = _date_rows(frame, date)
    counts = np.bincount(np.clip((rows['arrival_minute'].to_numpy() // 60).astype(int), 0, 23), minlength=24).astype(float)
    per_hour = steps // 24
    return np.clip(np.repeat(counts / float(ports) * float(growth) / per_hour, per_hour), 0.0, 1.0)


def day_profiles(days):
    """入札に入れるその日の確率（局 × 288）と、0時のEVを作るための前の2日の確率と日の種類。"""
    import numpy as np
    from EnvConfig import FUTURE_EV_ARRIVAL_GROWTH_MULTIPLIER as growth
    frames = _frames()
    probs, prev = [], []
    for d in days:
        info, frame = frames[d['source']]
        probs.append(hourly_probs(frame, d['date'], d['ports'], growth))
        prev.append([(hourly_probs(frame, p, d['ports'], growth), _class_of(frame, p)) for p in _prev_dates(d['date'])])
    return np.array(probs), prev


def probs_key(probs) -> str:
    """その日の到着の確率（局 × 288）を見分ける鍵。いろいろな日で学習するとき、reset に渡された確率から前の2日を引く。"""
    import hashlib
    import numpy as np
    return hashlib.sha1(np.round(np.asarray(probs, float), 12).tobytes()).hexdigest()


def _hourly_counts(frame, date):
    import numpy as np
    rows = _date_rows(frame, date)
    return np.bincount(np.clip((rows['arrival_minute'].to_numpy() // 60).astype(int), 0, 23), minlength=24).astype(float)


def forecast_counts(frame, date, cls):
    """前日の昼の予測（直近4週の同じ種類の日の平均）：date − 28日 〜 date − 2日の、日の種類が cls の記録のある日の、
    1時間ごとの件数の平均。forecast_eval.py で、4つのデータ源のどれでもほぼ一番外れが小さかった作り方。"""
    import datetime as dt
    import numpy as np
    D = dt.date.fromisoformat(date)
    per = {d: str(rows['day_class'].iloc[0]) for d, rows in frame.groupby('local_date')}
    days = [d for d, c in per.items() if c == cls and D - dt.timedelta(days=28) <= dt.date.fromisoformat(d) <= D - dt.timedelta(days=2)]
    if not days:
        raise ValueError(f'{date} の予測に使える日がない（{cls}）')
    return np.mean([_hourly_counts(frame, d) for d in days], axis=0), len(days)


# 機械学習の予測の水準：予測 = 単純な予測 + ALPHA_ML ×（実際 − 単純な予測）。外れは単純な予測の (1 − ALPHA_ML) 倍になる。
# JPL の1日の台数の WAPE が、手元の単純な予測（直近4週の平日の平均）で16.2%［実測、forecast_eval.json］、
# 機械学習（Nespoli ら 2023、7日先、掲載28日分から再計算）で11.3%［外部、原文を確認中］なので、1 − 11.3/16.2 ≒ 0.30。
# 他のデータ源にも同じ割合で当てはめる［推定］。
ALPHA_ML = 0.30


def forecast_profiles(days, alpha=0.0):
    """入札と学習の世界に使う、前日の昼の予測から作った到着の確率（局 × 288）と、0時のEVを流す前の2日の確率。
    その日と前日は forecast_counts（alpha > 0 なら、実際の件数の側へ alpha の割合だけ寄せる）、2日前は実際の件数
    （前日の昼には分かっている）。式は day_profiles と同じ。"""
    import numpy as np
    from EnvConfig import FUTURE_EV_ARRIVAL_GROWTH_MULTIPLIER as growth
    frames = _frames()
    per_hour = 288 // 24

    def to_probs(counts, ports):
        return np.clip(np.repeat(np.asarray(counts, float) / float(ports) * float(growth) / per_hour, per_hour), 0.0, 1.0)

    probs, prev, used = [], [], []
    for d in days:
        info, frame = frames[d['source']]
        counts, n = forecast_counts(frame, d['date'], _class_of(frame, d['date']))
        counts = counts + alpha * (_hourly_counts(frame, d['date']) - counts)
        probs.append(to_probs(counts, d['ports']))
        p2, p1 = _prev_dates(d['date'])
        c1, n1 = forecast_counts(frame, p1, _class_of(frame, p1))
        c1 = c1 + alpha * (_hourly_counts(frame, p1) - c1)
        prev.append([(to_probs(_hourly_counts(frame, p2), d['ports']), _class_of(frame, p2)),
                     (to_probs(c1, d['ports']), _class_of(frame, p1))])
        used.append({'station': d['station'], 'date': d['date'], 'forecast_total': float(counts.sum()), 'days_used': n,
                     'actual_total': float(_hourly_counts(frame, d['date']).sum())})
    return np.array(probs), prev, used


def install_preroll(prev, by_probs=None):
    """EVEnv._preroll_initial_evs を、前の2日のその日の確率で流す形にする（流し方は EVEnv と同じ）。
    by_probs（probs_key → 前の2日）を渡すと、reset に渡されたその日の確率から前の2日を選ぶ（見つからなければ止める）。"""
    from environment.EVEnv import EVEnv, POWER_TO_ENERGY

    def preroll(self, seed):
        steps = int(self.episode_steps)
        trial_rng = _r.Random(int(seed))
        attribute_seed = (int(seed) * 2654435761 + 97) % (2 ** 31)
        max_evs = int(self.max_ev_per_station)
        days_prev = prev
        if by_probs is not None:
            override = getattr(self, '_arrival_probability_override', None)
            key = probs_key(override) if override is not None else None
            if key not in by_probs:
                raise RuntimeError('reset に渡された到着の確率が、どの学習用の日とも合わない')
            days_prev = by_probs[key]
        present = []
        for st in range(self.num_stations):
            connected = []
            for day_offset, (probs, cls) in enumerate(days_prev[st]):
                first_step = (day_offset - len(days_prev[st])) * steps + 1
                for t in range(steps):
                    step_count = first_step + t
                    connected = [ev for ev in connected if ev['dep'] >= step_count]
                    p = float(probs[t])
                    if p <= 0.0:
                        continue
                    for slot_pos in range(max_evs):
                        if trial_rng.random() >= p or len(connected) >= max_evs:
                            continue
                        key = ((day_offset * steps + t) * self.num_stations + st) * max_evs + slot_pos
                        ev = self._session_attributes(st, cls, t, _r.Random(attribute_seed + key))
                        ev['arrival'] = step_count
                        ev['dep'] = step_count + ev['dwell_steps'] - 1
                        connected.append(ev)
            present.append(_at_midnight(connected, POWER_TO_ENERGY))
        return present

    original = EVEnv._preroll_initial_evs
    EVEnv._preroll_initial_evs = preroll
    return original


def _at_midnight(connected, power_to_energy):
    midnight = []
    for ev in connected:
        if ev['dep'] < 1:
            continue
        charged_steps = 1 - ev['arrival']
        gained = charged_steps * ev['max_power_kw'] * float(power_to_energy) * 100.0 / max(ev['capacity_kwh'], 1e-6)
        ev['init_soc'] = float(max(min(ev['init_soc'] + gained, ev['target_soc']), ev['init_soc']))
        midnight.append(ev)
    return midnight


def _real_event(env, st, cls, idx, rng):
    """プールの idx 番のセッション（実際の滞在時間と電力量）から、EVEnv._session_attributes と同じ形のEVを作る。"""
    import numpy as np
    from environment.EVEnv import (EV_TARGET_REACHABLE_POWER_FRACTION, EV_TARGET_SOC_MAX_PCT, POWER_TO_ENERGY,
                                   TIME_STEP_MINUTES)
    sessions = env.station_pools[int(st)].pools[cls]
    capacity_kwh, max_power_kw = env._sample_physical_profile(rng)
    soc = env.plugin_soc_bins.draw(rng, home=env.station_is_residential[int(st)])
    init_soc = float(np.clip(soc, 0.0, 100.0))
    dwell_steps = max(1, int(round(float(sessions.dwell_minutes[idx]) / float(TIME_STEP_MINUTES))))
    soc_per_kwh = 100.0 / max(float(capacity_kwh), 1e-6)
    delivered = max(float(sessions.energy_kwh[idx]), 0.0) * soc_per_kwh
    reachable = init_soc + float(dwell_steps) * float(max_power_kw) * float(POWER_TO_ENERGY) * soc_per_kwh * float(
        EV_TARGET_REACHABLE_POWER_FRACTION)
    target = min(init_soc + delivered, reachable, float(EV_TARGET_SOC_MAX_PCT))
    return {'init_soc': init_soc, 'target_soc': float(target), 'dwell_steps': int(dwell_steps),
            'needed_soc': float(target - init_soc), 'profile_ev_id': None,
            'capacity_kwh': float(capacity_kwh), 'max_power_kw': float(max_power_kw)}


def _pool_index(frame, cls):
    """データ源の表の行番号 → プール（その日の種類の行だけを表の順に並べたもの）の番号。"""
    rows = frame.index[frame['day_class'] == cls]
    return {int(r): i for i, r in enumerate(rows)}


def real_realization(env, days):
    """本当の状況：(到着の一覧, 0時にいるEV)。env は reset の後の EVEnv（中身の分布と口数に使う）。"""
    from environment.EVEnv import POWER_TO_ENERGY
    from EnvConfig import FUTURE_EV_ARRIVAL_GROWTH_MULTIPLIER as growth
    frames = _frames()
    steps = int(env.episode_steps)
    max_evs = int(env.max_ev_per_station)
    events = [[[None] * max_evs for _ in range(env.num_stations)] for _ in range(steps)]
    initial = []
    for d in days:
        st = int(d['station'])
        info, frame = frames[d['source']]
        f = max_evs * float(growth) / float(d['ports'])
        rng = _r.Random(REAL_SEED * 1000 + st)

        def copies():
            return int(math.floor(f)) + (1 if rng.random() < f - math.floor(f) else 0)

        # 0時にいるEV：前の2日の実際のセッションを、充電口の数の制限つきで流す
        connected = []
        for day_offset, p in enumerate(_prev_dates(d['date'])):
            cls = _class_of(frame, p)
            pidx = _pool_index(frame, cls)
            first_step = (day_offset - PREV_DAYS) * steps + 1
            arrivals = []
            for r, row in _date_rows(frame, p).iterrows():
                for _ in range(copies()):
                    arrivals.append((int(row['arrival_minute']) // 5, r))
            arrivals.sort()
            k = 0
            for t in range(steps):
                step_count = first_step + t
                connected = [ev for ev in connected if ev['dep'] >= step_count]
                while k < len(arrivals) and arrivals[k][0] == t:
                    if len(connected) < max_evs:
                        ev = _real_event(env, st, cls, pidx[arrivals[k][1]], rng)
                        ev['arrival'] = step_count
                        ev['dep'] = step_count + ev['dwell_steps'] - 1
                        connected.append(ev)
                    k += 1
        initial.append(_at_midnight(connected, POWER_TO_ENERGY))
        # その日の到着
        cls = _class_of(frame, d['date'])
        pidx = _pool_index(frame, cls)
        for r, row in _date_rows(frame, d['date']).iterrows():
            t = min(int(row['arrival_minute']) // 5, steps - 1)
            for _ in range(copies()):
                slot = next((j for j in range(max_evs) if events[t][st][j] is None), None)
                if slot is not None:
                    events[t][st][slot] = _real_event(env, st, cls, pidx[r], rng)
    return events, initial


class fixed_day:
    """with fixed_day(events, initial): の中で作る EVEnv の1日を、決めた到着の一覧と0時のEVにする（本当の状況の評価用）。"""

    def __init__(self, events, initial):
        self.events, self.initial = events, initial

    def __enter__(self):
        from environment.EVEnv import EVEnv
        self.orig = (EVEnv._pregenerate_arrival_events, EVEnv._preroll_initial_evs)
        orig_events, orig_initial = self.orig
        events, initial = self.events, self.initial

        def ev(env):
            orig_events(env)  # 環境の乱数の流れを本線と同じに進める
            return copy.deepcopy(events)

        def ini(env, seed):
            return copy.deepcopy(initial)

        EVEnv._pregenerate_arrival_events, EVEnv._preroll_initial_evs = ev, ini
        return self

    def __exit__(self, *exc):
        from environment.EVEnv import EVEnv
        EVEnv._pregenerate_arrival_events, EVEnv._preroll_initial_evs = self.orig
        return False


def main():
    world.configure(TASK)
    if sys.argv[1:] == ['more']:
        days = choose_more_days()
        for name, rows in days.items():
            print(name, flush=True)
            for d in rows:
                print(f"  局{d['station']} {d['source'][:22]:22s} {d['date']} {d['sessions']:3d}件（平日の中央値 {d['median_sessions']:.0f}件、"
                      f"{d['rank'] + 1}/{d['weekdays']}番目、狙い {d['target_rank'] + 1:.1f}）", flush=True)
        (HERE / 'days_more.json').write_text(json.dumps(days, ensure_ascii=False, indent=1), encoding='utf-8')
        return
    days = choose_days()
    for name, rows in days.items():
        print(name, flush=True)
        for d in rows:
            print(f"  局{d['station']} {d['source'][:22]:22s} {d['date']} {d['sessions']:3d}件（平日の中央値 {d['median_sessions']:.0f}件、"
                  f"{d['rank'] + 1}/{d['weekdays']}番目）", flush=True)
    (HERE / 'days.json').write_text(json.dumps(days, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
