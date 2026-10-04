# 収束済み下位制御器に対する fine-tune が効かない根拠

2026-09-17 記録。実測部分は `execute_results/ft_sweep/` のログと
`archive/ft_a*/` の TensorBoard から取った値。文献部分は本文を読んで条件を確認した。

---

## 1. 実測：5構成すべてで zero-shot を超えなかった

事前学習は 7station・256指令・25日、ep1628 で収束。fine-tune は固定入札1日分
（2024-12-04）に対して 100 エピソード、指令は入札に添付された `feedback`
パーティション（入札設計に使った `forecast` とは別）。

| arm | 条件 | 選択時の追従（warm start = 95.32%） |
|---|---|---|
| a1 | baseline | ep10 95.40 / ep50 95.15 / ep100 95.20 |
| a2 | オフラインデータ併用なし | ep10 95.12 / ep50 94.96 / ep100 94.96 → 該当なし |
| a3 | 学習率 ×10 | ep10 95.04 / ep50 94.99 / ep100 95.02 → 該当なし |
| a5 | 報酬を補正後出力に変更 | TEST10 で raw 74.96%、途中停止 |
| a6 | critic ヘッド全体をリセット | TEST10 で raw 58.01%、崩壊、途中停止 |

a1 の ep10 が +0.08pt で唯一 warm start を上回ったが、選択は1シードなので
区別できない。**実質的に5構成すべてで zero-shot が最良。**

### 学習自体は成功している

TEST 曲線（ep10 → ep100）:

| arm | raw 追従 | 補正後追従 | raw MAE | 補正後 MAE | 離脱不足 |
|---|---|---|---|---|---|
| a1 | 82.98 → 88.23 (+5.25) | 93.48 → 92.70 (−0.78) | 58.9 → 45.0 | 15.0 → 16.3 | 8.03 → 9.33 |
| a2 | 82.55 → 87.23 (+4.68) | 92.91 → 93.19 (+0.28) | 59.8 → 41.8 | 15.4 → 15.1 | 15.44 → 12.29 |
| a3 | 84.96 → 88.65 (+3.69) | 93.19 → 92.84 (−0.35) | 56.4 → 44.3 | 15.8 → 16.2 | 9.68 → 11.62 |

学習器は自分の目的関数上では3本とも確実に改善している（raw +3.7〜5.3pt、
MAE −20〜30%）。にもかかわらず評価量は動かないか下がる。アーム内平均を引いて
プールした相関は **r(raw追従, 補正後追従) = −0.54**（n=30）。
「学習が進まない」のではなく、**進んだ分が補正後に届いていない。**

### local/global 勾配の和解が壊れやすい

`GradHealth/actor_source_local_global_cos`（actor に入る local 勾配と global 勾配の余弦）:

| run | cos 序盤 → 終盤 | global 勾配のノルム比 |
|---|---|---|
| pretrain（1628ep） | −0.461 → −0.157 | 0.584 → 0.795 |
| a1 baseline | −0.090 → −0.020 | 0.778 |
| a3 lr×10 | +0.154 → −0.077 | 0.938 |
| a6 critic reset | −0.479 → **−0.748** | 0.513 |

事前学習は1628エピソードかけて、対立していた2つの勾配（−0.46）をほぼ直交
（−0.16）まで和解させている。a6 が崩壊したのは学習率でも忘却でもなく、
**リセットがこの和解を壊した**ため。critic を動かす手法はここを壊した時点で負ける。

---

## 2. 文献：成功例はすべて前提が違う

| 手法 | 事前学習 | オンライン相互作用 | UTD | 開始 → 到達 |
|---|---|---|---|---|
| Balanced Replay (Lee+ 2021) | オフライン CQL×5、D4RL 100万〜200万遷移 | 25万step（操作系は約4〜8万step） | 記載なし | halfcheetah-medium 約4500リターン起点。終点は曲線のみで数値記載なし |
| Cal-QL (Nakamoto+ 2023) | オフライン100万step | **100万step**（Kitchen 125万） | 1（高UTD版 20） | large-diverse 25→87、door 22→88、relocate 6→98、平均 +106.9% |
| WSRL (Zhou+ 2024) | オフライン（Cal-QL等）100万step | 50万〜100万step | 4（Q関数10個） | AntMaze 0%付近 → 70〜80%+ |
| PEX (Zhang+ 2023) | オフライン IQL 100万step | 100万step | — | 集約曲線のみ |
| JSRL (Uchendu+ 2022) | BC or オフライン IQL、デモ20〜2万本 | 100万step（把持10万step） | — | antmaze-umaze 0.2 → 71.7 |
| Resets (Nikishin+ 2022) | スクラッチ | — | **高 replay ratio ほど効く** | リセット周期 2×10⁵ step |

### 本研究との差

| | 事前学習 | オンライン相互作用 | UTD | 開始性能 |
|---|---|---|---|---|
| 本研究 | **オンラインで収束済み**（ep1628） | **2.88万step**（100ep×288） | **1** | 追従 **95.32%** |

差は3点、どれも同じ方向を向く。

1. **事前学習がオフラインではない。** 上記5本は、オフラインデータから学んだ
   Q 関数の悲観性・OOD 外挿という病理を治すための手法。Balanced Replay の配合、
   Cal-QL の較正、WSRL の warmup は全部その治療。本研究はオンラインで収束させて
   いるのでその病理を持たない。効かないのは想定内。
2. **相互作用量が1〜2桁足りない。** 最短の Cal-QL 視覚操作でも10万step、多くは
   100万step。本研究は2.88万step。Nikishin のリセット周期（2×10⁵）ですら
   本研究の fine-tune 全体（2.88×10⁴）より1桁長い。
3. **埋めるべき差が違う。** 文献は 6→98、0.2→71.7 のように「失敗している方策を
   成功させる」。本研究は 95.32% から上を狙う。同じ道具の適用範囲ではない。

### 条件が近い報告

- **Zero-Shot RL from Low Quality Data**（arXiv:2309.15178）— オンライン
  fine-tune 中に事前学習（zero-shot）性能を一度も超えなかった、という報告。
  本研究の a1〜a3 で3本とも起きた現象と同じ。
- **Breaking the Performance Ceiling in RL requires Inference Strategies**
  （arXiv:2505.21236）— 潜在空間の CMA-ES 探索がオンライン fine-tune より良く、
  勾配降下より局所解にはまりにくい、と報告。方策パラメータを勾配で動かすのを
  やめ、推論時に探索する方向。

---

## 3. 結論

**fine-tune が効かないことは失敗ではなく、条件から予測できる結果。**
オンラインで収束した下位制御器は、その日の入札に対して追加学習なしで機能する。
系統的に試した5構成はいずれも改善しなかった。zero-shot が最良であること自体が
測定結果であり、報告できる。

## 出典

- Lee et al., Offline-to-Online RL via Balanced Replay and Pessimistic Q-Ensemble, CoRL 2021. arXiv:2107.00591
- Nakamoto et al., Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning, 2023. arXiv:2303.05479
- Zhou et al., Efficient Online RL Fine-Tuning Need Not Retain Offline Data (WSRL), 2024. arXiv:2412.07762
- Zhang et al., Policy Expansion for Bridging Offline-to-Online RL (PEX), ICLR 2023. arXiv:2302.00935
- Uchendu et al., Jump-Start Reinforcement Learning (JSRL), 2022. arXiv:2204.02372
- Nikishin et al., The Primacy Bias in Deep Reinforcement Learning, ICML 2022. arXiv:2205.07802
- Zero-Shot Reinforcement Learning from Low Quality Data. arXiv:2309.15178
- Breaking the Performance Ceiling in Reinforcement Learning requires Inference Strategies. arXiv:2505.21236
