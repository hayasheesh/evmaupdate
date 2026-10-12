"""保存済みbest/late NPZだけを読み、EV属性別の出発時SoCを再集計する。"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
THRESHOLD_KWH = 0.001
DT = 5 / 60


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def band(value, edges, labels):
    return labels[int(np.searchsorted(edges, value, side="right"))]


def stats(records):
    n = len(records)
    missed = [r for r in records if r["miss"]]
    lead = [r["unrecoverable_lead_minutes"] for r in missed
            if r["unrecoverable_lead_minutes"] is not None]
    def mean(key):
        return float(np.mean([r[key] for r in records])) if n else None
    return {
        "EV_count": n, "miss_count": len(missed),
        "miss_rate_pct": 100 * len(missed) / n if n else None,
        "deficit_kwh_sum": sum(r["deficit_kwh"] for r in records),
        "miss_deficit_kwh_sum": sum(r["deficit_kwh"] for r in missed),
        "initial_count": sum(r["initial"] for r in records),
        "last_hour_discharge_EV_count": sum(r["last_hour_discharge_kwh"] > 1e-6 for r in records),
        "miss_with_last_hour_discharge_count": sum(r["last_hour_discharge_kwh"] > 1e-6 for r in missed),
        "last_hour_discharge_kwh_sum": sum(r["last_hour_discharge_kwh"] for r in records),
        "miss_last_hour_discharge_kwh_sum": sum(r["last_hour_discharge_kwh"] for r in missed),
        "miss_last_hour_discharge_below_target_count": sum(r["last_hour_discharge_below_target_kwh"] > 1e-6 for r in missed),
        "last_hour_observation_shorter_than_1h_count": sum(r["observed_hours"] < 1 for r in records),
        "miss_first_unrecoverable_action_discharge_count": sum(r["first_unrecoverable_action_discharge"] is True for r in missed),
        "miss_initially_unrecoverable_count": sum(r["initial_margin_kwh"] < -THRESHOLD_KWH for r in missed),
        "miss_became_unrecoverable_count": sum(r["initial_margin_kwh"] >= -THRESHOLD_KWH and r["unrecoverable_lead_minutes"] is not None for r in missed),
        "miss_without_recorded_unrecoverability_count": len(missed) - len(lead),
        "miss_lead_denominator": len(lead),
        "miss_lead_minutes_median": float(np.median(lead)) if lead else None,
        "miss_lead_minutes_p25_p75": np.percentile(lead, [25, 75]).tolist() if lead else None,
        "observed_hours_mean": mean("observed_hours"),
        "first_soc_mean": mean("first_soc"),
        "required_kwh_mean": mean("required_kwh"),
        "capacity_kwh_mean": mean("capacity_kwh"), "rated_kw_mean": mean("rated_kw"),
    }


AXES = {
    "origin": "初期在車か当日到着か",
    "observed_duration": "初期在車は開始時の残り時間、当日到着は日内で観測した滞在時間",
    "arrival_soc": "当日到着時SoC。初期在車は開始時SoCであり到着時SoCは不明",
    "required_energy": "最初の観測時に目標まで必要なkWh",
    "capacity": "容量kWh", "rating": "定格kW",
    "deadline": "評価時刻。288ステップは実出発と日末評価を区別できない",
    "last_hour_discharge": "出発・評価までの最後の12ステップでの放電有無",
    "unrecoverable_lead": "最初に回復余裕が-0.001kWhを下回った行の行動後から評価までの分数",
    "initial_slack": "最初の観測時の定格充電可能量から必要量を引いたkWh",
    "origin_deadline": "初期在車・当日到着と評価時刻の組み合わせ",
}


def cohorts(records):
    result = {}
    for axis in AXES:
        groups = defaultdict(list)
        for r in records:
            groups[r["cohorts"][axis]].append(r)
        result[axis] = {k: dict(stats(v), composition_pct=100 * len(v) / len(records))
                        for k, v in sorted(groups.items())}
    return result


def read_case(path, columns, case):
    with np.load(path, allow_pickle=False) as z:
        e = z["ev"].astype(np.float64)
        steps = z["step"]
        assert len(steps) == 288 and steps[-1, 0] == 288
    c = {name: i for i, name in enumerate(columns)}
    def col(name):
        return e[:, c[name]]
    keys = np.stack([col("station"), col("id"), col("arrival_step")], axis=1).astype(np.int64)
    order = np.lexsort((col("step"), keys[:, 2], keys[:, 1], keys[:, 0]))
    e, keys = e[order], keys[order]
    cuts = np.r_[0, np.flatnonzero(np.any(keys[1:] != keys[:-1], axis=1)) + 1, len(e)]
    records = []
    for start, stop in zip(cuts[:-1], cuts[1:]):
        rows = e[start:stop]
        f, last = rows[0], rows[-1]
        def get(row, name):
            return float(row[c[name]])
        rem = rows[:, c["remaining_steps"]]
        t = rows[:, c["step"]]
        assert np.all(np.diff(t) == 1)
        assert np.all(t + rem - 1 == t[0] + rem[0] - 1)
        assert np.all(rows[:, [c[x] for x in ("target", "capacity_kwh", "rated_kw")]] ==
                      f[[c[x] for x in ("target", "capacity_kwh", "rated_kw")]])
        assert get(last, "remaining_steps") == 1, "全EVの評価行を要する"
        assert len(rows) == int(get(f, "remaining_steps"))
        cap, power, soc, target = [get(f, x) for x in ("capacity_kwh", "rated_kw", "soc_before", "target")]
        initial = get(f, "arrival_step") == 0
        assert get(f, "step") == (1 if initial else get(f, "arrival_step"))
        need = max(0, (target - soc) * cap / 100)
        margin_before = rem * power * DT - np.maximum((target - rows[:, c["soc_before"]]) * cap / 100, 0)
        margin_after = rows[:, c["recoverable_margin_after_kwh"]]
        # 記録された余裕と列から再計算した余裕の整合性を検査する。
        recalculated = (rem - 1) * power * DT - np.maximum((target - rows[:, c["soc_after"]]) * cap / 100, 0)
        assert np.max(np.abs(margin_after - recalculated)) < 0.003
        bad = np.flatnonzero(margin_after < -THRESHOLD_KWH)
        lead = float((rem[bad[0]] - 1) * 5) if len(bad) else None
        deficit = max(0, (target - get(last, "soc_after")) * cap / 100)
        last_hour = rem <= 12
        discharged = np.maximum(-rows[:, c["executed_kw"]], 0) * DT
        below = (target - rows[:, c["soc_before"]]) * cap / 100 > THRESHOLD_KWH
        duration = get(f, "remaining_steps") * DT
        deadline = int(get(last, "step"))
        r = {
            "case_id": case["id"], "date": case["date"], "command": case["command"], "ev_seed": case["ev_seed"],
            "station": int(get(f, "station")), "id": int(get(f, "id")), "arrival_step": int(get(f, "arrival_step")),
            "initial": initial, "first_step": int(get(f, "step")), "deadline_step": deadline,
            "observed_hours": duration, "first_soc": soc, "arrival_soc": None if initial else soc,
            "target_soc": target, "required_kwh": need, "capacity_kwh": cap, "rated_kw": power,
            "initial_margin_kwh": float(margin_before[0]), "final_soc": get(last, "soc_after"),
            "deficit_kwh": deficit, "miss": deficit > THRESHOLD_KWH,
            "raw_strict_miss": get(last, "soc_after") < target,
            "last_hour_discharge_kwh": float(discharged[last_hour].sum()),
            "last_hour_discharge_below_target_kwh": float(discharged[last_hour & below].sum()),
            "day_final_hour_discharge_kwh": float(discharged[t >= 277].sum()),
            "unrecoverable_first_step": int(t[bad[0]]) if len(bad) else None,
            "unrecoverable_lead_minutes": lead,
            "first_unrecoverable_action_discharge": bool(rows[bad[0], c["executed_kw"]] < -1e-6) if len(bad) else None,
        }
        r["cohorts"] = {
            "origin": "初期在車" if initial else "当日到着",
            "observed_duration": ("初期残り時間" if initial else "当日日内滞在") + ":" + band(duration, [1, 3, 6, 12], ["1h未満", "1–3h", "3–6h", "6–12h", "12h以上"]),
            "arrival_soc": ("開始時SoC" if initial else "到着時SoC") + ":" + band(soc, [20, 40, 60, 80], ["20%未満", "20–40%", "40–60%", "60–80%", "80%以上"]),
            "required_energy": band(need, [0.001, 10, 30, 60], ["0.001kWh未満", "0.001–10kWh", "10–30kWh", "30–60kWh", "60kWh以上"]),
            "capacity": band(cap, [50, 80, 100], ["50kWh未満", "50–80kWh", "80–100kWh", "100kWh以上"]),
            "rating": band(power, [10, 25, 50], ["10kW未満", "10–25kW", "25–50kW", "50kW以上"]),
            "deadline": "日末境界" if deadline == 288 else band(deadline * DT, [6, 12, 18], ["00–06時", "06–12時", "12–18時", "18–24時"]),
            "last_hour_discharge": "放電あり" if r["last_hour_discharge_kwh"] > 1e-6 else "放電なし",
            "unrecoverable_lead": "記録なし" if lead is None else band(lead, [5, 15, 60, 180], ["0分", "5–15分未満", "15–60分未満", "60–180分未満", "180分以上"]),
            "initial_slack": band(r["initial_margin_kwh"], [-0.001, 1, 10, 50], ["初めから回復不能", "-0.001–1kWh", "1–10kWh", "10–50kWh", "50kWh以上"]),
        }
        r["cohorts"]["origin_deadline"] = r["cohorts"]["origin"] + ":" + r["cohorts"]["deadline"]
        records.append(r)
    assert len(records) == case["depart_count"]
    return records


def key(r):
    return r["case_id"], r["station"], r["id"], r["arrival_step"]


def compare(best, late):
    b, l = {key(r): r for r in best}, {key(r): r for r in late}
    assert b.keys() == l.keys()
    attrs = ["date", "ev_seed", "initial", "first_soc", "target_soc", "required_kwh", "capacity_kwh", "rated_kw", "observed_hours", "deadline_step"]
    assert all(all(b[k][a] == l[k][a] for a in attrs) for k in b)
    def paired_stats(pairs):
        n = len(pairs)
        return {
            "paired_EV_count": n,
            "both_pass": sum(not x["miss"] and not y["miss"] for x, y in pairs),
            "both_miss": sum(x["miss"] and y["miss"] for x, y in pairs),
            "best_pass_late_miss": sum(not x["miss"] and y["miss"] for x, y in pairs),
            "best_miss_late_pass": sum(x["miss"] and not y["miss"] for x, y in pairs),
            "late_minus_best_deficit_kwh": sum(y["deficit_kwh"] - x["deficit_kwh"] for x, y in pairs),
            "late_minus_best_last_hour_discharge_kwh": sum(y["last_hour_discharge_kwh"] - x["last_hour_discharge_kwh"] for x, y in pairs),
        }
    pairs = [(b[k], l[k]) for k in sorted(b)]
    grouped = {}
    # 行動後に決まる属性で構成を調整せず、観測開始時の属性を使う。
    for axis in list(AXES)[:7] + ["initial_slack", "origin_deadline", "station"]:
        groups = defaultdict(list)
        for x, y in pairs:
            label = str(x["station"] + 1) if axis == "station" else x["cohorts"][axis]
            groups[label].append((x, y))
        grouped[axis] = {k: paired_stats(v) for k, v in sorted(groups.items())}
    return dict(paired_stats(pairs), same_EV_attributes=True, strata=grouped)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", "--input-base", dest="input_root", type=Path, default=ROOT,
                        help="metadata/summaryのある単一市場フォルダ、または市場フォルダの親")
    parser.add_argument("--input-pattern", default="trace_plateau20_{market}_20261008",
                        help="親フォルダ指定時の市場フォルダ名。{market}を使える")
    parser.add_argument("--folder-prefix", help="例 calendar20。指定時はinput-patternより優先")
    parser.add_argument("--folder-suffix", default="20261008", help="市場名に続くフォルダ名の日付部分")
    parser.add_argument("--markets", nargs="+", choices=["aemo", "ercot"], default=["aemo", "ercot"])
    parser.add_argument("--output", type=Path, default=ROOT / "EV_cohort_analysis.json")
    args = parser.parse_args()
    if (args.input_root / "metadata.json").is_file():
        single_meta = json.loads((args.input_root / "metadata.json").read_text(encoding="utf-8"))
        folders = {single_meta["market"]: args.input_root}
    else:
        pattern = (args.folder_prefix + "_{market}_" + args.folder_suffix) if args.folder_prefix else args.input_pattern
        folders = {market: args.input_root / pattern.format(market=market) for market in args.markets}
    output = {
        "source": "[raw再集計]", "diagnostic_threshold_kwh": THRESHOLD_KWH,
        "analysis_script_sha256": digest(Path(__file__)),
        "definitions_ja": {
            "scope": "各市場のevaluation_scopeに入力から求めた日数・ケース数・指令数・EVseed数を記録する。保存済みbest/lateのみ。",
            "miss": "出発・評価行remaining_steps=1でmax(target-soc_after,0)×capacity/100 > 0.001kWh。厳密な公式metricとは異なる診断閾値。",
            "official_metric": "summaryのsocは厳密比較によるケース別公式達成率の平均。rawのsoc_afterはfloat32で保存され、丸めも含むため公式metricの再現とは扱わない。",
            "initial": "arrival_step=0。開始時SoCと残り時間を集計する。実際の到着SoCと全滞在時間は不明。",
            "boundary": "[コード確認]EVEnv._horizon_obligationは日末を超える出発を288へ変更して日末目標を定める。NPZは元の出発時刻を保持せず、288での実出発と日末評価を区別できない。",
            "recovery": "行動後のrecoverable_margin_after_kwh < -0.001。以降の全ステップを定格充電しても記録された目標に届かない診断。全体の指令とEVを初めから知るLPの実行可能性とは異なる。",
            "lead": "(初回回復不能行のremaining_steps-1)×5分。行動後から評価までの時間。初めから回復不能なEVは別集計。",
            "last_hour": "remaining_steps<=12でexecuted_kw<0のエネルギー。12ステップ未満の滞在は観測した全期間。day_final_hourはstep>=277。",
            "bins": "数値の区間は下端を含み上端を含まない。0分の区間は0以上5未満。局番号はper_EV_recordsのみ0始まり、station集計は1始まり。",
            "composition": "各属性のEV数と全EVに占める割合を示す。best/lateは同一EV属性を照合するため、その差はEV構成差によらない。",
            "market_difference": "市場間の差には指令と行動器の双方の違いが含まれる。属性を照合しても行動器単独の差には分離できない。",
            "station_difference": "局ごとの属性と未達率を示す。局間では指令に対する観測と履歴も異なり、属性別未達率だけでは行動器単独の差を確定できない。",
            "association": "最終1hの放電と未達の同時発生は関連の集計。放電の因果効果は未測定。",
        },
        "cohort_axes_ja": AXES, "markets": {},
    }
    all_records = {}
    for market, folder in folders.items():
        metadata_path, summary_path = folder / "metadata.json", folder / "summary.json"
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert meta["market"] == market
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        result = {"metadata": meta, "input_sha256": {"metadata.json": digest(metadata_path), "summary.json": digest(summary_path)}, "modes": {}}
        selected = {mode: sorted([r for r in summary["rows"] if r["mode"] == mode], key=lambda r: r["id"]) for mode in ("best", "late")}
        assert selected["best"] and len(selected["best"]) == len(selected["late"])
        assert len({r["id"] for r in selected["best"]}) == len(selected["best"])
        dates = sorted({r["date"] for r in selected["best"]})
        scope = {"input_root": str(folder.resolve()), "metadata_scope": meta.get("scope"),
                 "date_count": len(dates), "case_count_per_mode": len(selected["best"]),
                 "mode_count": 2, "station_count": len(meta.get("actor_sha256", {}).get(str(meta.get("best_episode")), [])) or None,
                 "dates": dates, "per_date": {}}
        for date in dates:
            daily = [r for r in selected["best"] if r["date"] == date]
            scope["per_date"][date] = {"case_count": len(daily),
                                      "command_count": len({r["command"] for r in daily}),
                                      "EV_seed_count": len({r["ev_seed"] for r in daily}),
                                      "commands": sorted({r["command"] for r in daily}),
                                      "EV_seeds": sorted({r["ev_seed"] for r in daily})}
        result["evaluation_scope"] = scope
        for mode, cases in selected.items():
            records, case_stats = [], []
            for case in cases:
                path = folder / f"{mode}_{case['id']}.npz"
                evs = read_case(path, meta["ev_columns"], case)
                records.extend(evs)
                case_stats.append(dict(stats(evs), case_id=case["id"], date=case["date"], command=case["command"], ev_seed=case["ev_seed"], ev_hash=case["ev_hash"], official_soc_pct=case["soc"]))
                result["input_sha256"][path.name] = digest(path)
            overall = stats(records)
            overall["raw_strict_miss_count_not_official"] = sum(r["raw_strict_miss"] for r in records)
            overall["day_final_hour_discharge_kwh_sum"] = sum(r["day_final_hour_discharge_kwh"] for r in records)
            overall["unrecoverability_first_action_discharge_count"] = sum(r["first_unrecoverable_action_discharge"] is True for r in records)
            overall["official_soc_case_mean_pct"] = summary["summary"][mode]["soc"]
            overall["diagnostic_soc_case_mean_pct"] = 100 - float(np.mean([r["miss_rate_pct"] for r in case_stats]))
            result["modes"][mode] = {
                "overall": overall, "cases": case_stats, "cohorts": cohorts(records),
                "stations": {str(st + 1): dict(stats([r for r in records if r["station"] == st]), cohorts=cohorts([r for r in records if r["station"] == st])) for st in range(20)},
                "per_EV_records": records,
            }
            all_records[market, mode] = records
        assert all(b["ev_hash"] == l["ev_hash"] and b["command"] == l["command"] for b, l in zip(selected["best"], selected["late"]))
        result["best_to_late"] = compare(all_records[market, "best"], all_records[market, "late"])
        result["report_ja"] = [f"[raw再集計] {market.upper()}の{len(dates)}日、モデルごと{len(selected['best'])}ケース。"]
        for mode in ("best", "late"):
            values = result["modes"][mode]
            total = values["overall"]
            result["report_ja"].append(
                f"[raw再集計] {mode}の実質未達は{total['miss_count']}/{total['EV_count']}台（{total['miss_rate_pct']:.2f}%）。"
                f"未達EVの最終1h放電ありは{total['miss_with_last_hour_discharge_count']}/{total['miss_count']}台。"
                f"回復不能化の先行時間中央値は{total['miss_lead_minutes_median']}分、分母{total['miss_lead_denominator']}台。")
            for axis, grouped in values["cohorts"].items():
                parts = [f"{name} {v['miss_count']}/{v['EV_count']}台（{v['miss_rate_pct']:.2f}%）、構成比{v['composition_pct']:.2f}%"
                         for name, v in grouped.items()]
                result["report_ja"].append(f"[raw再集計] {mode} {AXES[axis]}。" + "。".join(parts) + "。")
        paired = result["best_to_late"]
        result["report_ja"].append(
            f"[raw再集計] 同一属性を照合した{paired['paired_EV_count']}台のうち、"
            f"best未達からlate達成は{paired['best_miss_late_pass']}台、"
            f"best達成からlate未達は{paired['best_pass_late_miss']}台。EV構成は一致した。")
        output["markets"][market] = result
    output["market_EV_composition_comparison"] = {}
    for mode in (("best", "late") if set(folders) == {"aemo", "ercot"} else ()):
        matched = compare(all_records["aemo", mode], all_records["ercot", mode])
        # compareの名称best/lateはここではAEMO/ERCOTの順に対応する。
        output["market_EV_composition_comparison"][mode] = {
            "same_EV_attributes": matched["same_EV_attributes"], "paired_EV_count": matched["paired_EV_count"],
            "aemo_pass_ercot_miss": matched["best_pass_late_miss"],
            "aemo_miss_ercot_pass": matched["best_miss_late_pass"],
            "both_miss": matched["both_miss"], "both_pass": matched["both_pass"],
        }
    path = args.output
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    for market, result in output["markets"].items():
        for mode, values in result["modes"].items():
            print(market, mode, json.dumps(values["overall"], ensure_ascii=False))
        print(market, "best_to_late", json.dumps({k: v for k, v in result["best_to_late"].items() if k != "strata"}, ensure_ascii=False))
    print("市場間EV構成", json.dumps(output["market_EV_composition_comparison"], ensure_ascii=False))


if __name__ == "__main__":
    main()
