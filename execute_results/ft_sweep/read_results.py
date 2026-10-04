"""各アームの selection・ペア評価・中間テスト曲線をまとめて読む。"""
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAT = re.compile(
    r"\[TEST(\d+) summary\].*?global=([\d.]+).*?SoC actor/physical=([\d.]+)/"
    r".*?deficit=([\d.]+)kWh \| MARL/Central=([\d.]+)/([\d.]+)%.*?MAE=([\d.]+)->([\d.]+)kW"
)


def curves():
    out = {}
    for log in sorted((ROOT / "execute_results" / "ft_sweep").glob("*.log")):
        rows = [
            {
                "ep": int(m.group(1)), "報酬": float(m.group(2)),
                "MARL": float(m.group(5)), "査定": float(m.group(6)),
                "生MAE": float(m.group(7)), "後MAE": float(m.group(8)),
                "SoC": float(m.group(3)), "不足": float(m.group(4)),
            }
            for m in PAT.finditer(log.read_text(encoding="utf-8", errors="replace"))
        ]
        if rows:
            out[log.stem] = rows
    return out


def summaries():
    out = {}
    for d in sorted((ROOT / "archive").glob("ft_*")):
        f = d / "results" / "single_day_finetune_summary.json"
        if f.is_file():
            out[d.name] = json.loads(f.read_text(encoding="utf-8"))
    return out


if __name__ == "__main__":
    cs = curves()
    for arm, rows in cs.items():
        print(f"\n### {arm}  中間テスト (holdout 24指令)")
        print(f"{'ep':>4} {'報酬':>7} {'MARL':>7} {'査定':>7} {'生MAE':>7} {'後MAE':>7} {'SoC':>7} {'不足':>7}")
        for r in rows:
            print(f"{r['ep']:>4} {r['報酬']:>7.4f} {r['MARL']:>7.2f} {r['査定']:>7.2f} "
                  f"{r['生MAE']:>7.1f} {r['後MAE']:>7.1f} {r['SoC']:>7.2f} {r['不足']:>7.2f}")
        if len(rows) >= 2:
            a, b = rows[0], rows[-1]
            print(f"  ep{a['ep']}→{b['ep']}:  報酬 {b['報酬']-a['報酬']:+.4f}   "
                  f"MARL {b['MARL']-a['MARL']:+.2f}pt   査定 {b['査定']-a['査定']:+.2f}pt   "
                  f"SoC {b['SoC']-a['SoC']:+.2f}pt   不足 {b['不足']-a['不足']:+.2f}kWh")

    for arm, d in summaries().items():
        s = d.get("selection") or {}
        print(f"\n### {arm}  選択とペア評価")
        print(f"  学習 {d.get('train_activation_scenarios')}({d.get('train_activation_partition')})"
              f"  評価 {d.get('eval_activation_scenarios')}({d.get('eval_activation_partition')})")
        for c in s.get("candidates") or []:
            mark = " ← 採用" if c["tag"] == s.get("adopted") else ""
            print(f"    {c['tag']:12s} 査定 {c['global_tracking_rate']*100:6.2f}%  "
                  f"SoC {c['soc_hit_rate']*100:6.2f}%{mark}")
        if "marl_only_tracking_before" in s:
            print(f"    MARL単独  warm start {s['marl_only_tracking_before']*100:.2f}% "
                  f"→ 採用 {s['marl_only_tracking_after']*100:.2f}%  "
                  f"(遮蔽={s.get('correction_masked_a_regression')})")
        ks = ("global_tracking_rate", "controller_pre_system_tracking_rate",
              "central_ev_tracking_rate", "post_bess_mae_kw", "soc_hit_rate", "up_pass_rate")
        if d.get("finetuned") and d.get("zeroshot"):
            print(f"    {'指標':<38}{'fine-tune後':>12}{'zero-shot':>12}{'差':>10}")
            for k in ks:
                fv, zv = d["finetuned"].get(k), d["zeroshot"].get(k)
                if fv is None or zv is None:
                    continue
                sc = 1.0 if "mae" in k else 100.0
                print(f"    {k:<38}{fv*sc:>12.2f}{zv*sc:>12.2f}{(fv-zv)*sc:>+10.2f}")
