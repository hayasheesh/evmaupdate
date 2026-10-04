# EVMA-LOCAL

EV-VPPがload-following型（二次調整力②相当）の需給調整商品へ参入できるかを、入札から当日制御まで通して評価する研究コード。

- 主題と貢献: [研究の全体像](docs/研究の全体像.md)
- 入札器: [入札器の問題設定](docs/入札器の問題設定.md)
- 指令: [指令データ](docs/指令データ.md)
- 市場の規定: [市場規定と海外事例](docs/市場規定と海外事例.md)
- 分散制御の根拠: [分散制御を使う理由](docs/分散制御を使う理由.md)

## 提案システム

1. 前日の予測分布と過去の実指令から、入札器が基準値・上げ幅・下げ幅を決める。
2. 当日はstation単位のMARLが、ローカル観測と市場指令からEV電力を決める。
3. force充電が、出発時SoCを達成不能にする行動だけを直す。
4. 連系点（PCC）のBESSが残った追従誤差を吸収する。

中央残差分配は提案システムに入れない。全stationが中央の目標に従う場合の比較器 `rule_based_central` としてだけ残す。

## 実行入口

指令集合は `EVMA_ACTIVATION_SIGNAL_SET` で選ぶ（既定は `aemo_plan_deviation`）。集合の中身は[指令データ](docs/指令データ.md)を参照。

入札bankの作成:

```powershell
python tools/build_all_upper_bid_banks.py --workers 2
```

`--preflight-only` を付けると、指令の区分と本数だけ確認して止まる。

MARLの事前学習:

```powershell
python pre_train.py
```

学習の仕方は `EVMA_MARL_ALGORITHM` で選ぶ。

- `hybrid`（既定）：本研究の学習法。stationごとのSoC用critic、全体の追従用critic（QMIX風の混合器）、2つの勾配を混ぜた行動器の更新、TD3式の工夫を持つ
- `maddpg`：Lowe et al.（2017）のMADDPG。論文で標準実装との差を示すための比較用。stationごとに1つのcriticが全stationの観測と行動を見て、そのstationの報酬（SoC用と追従用の和）を1つのγ（既定0.95）で学ぶ。行動器・観測・行動の制限・学習率などは `hybrid` と同じにしてある。詳細は `training/Agent/standard_maddpg.py`

提案システムの評価（`--pipeline` は `marl_raw`、`marl_force`、`marl_force_bess`、`rule_based_central`）:

```powershell
python tools/evaluate_final_system_on_bid_bank.py `
  --model-dir <run> --output-dir <output> `
  --pipeline marl_force_bess
```

`rule_based_central` でもcheckpointを読み込むが、観測次元を決めるためだけでactorの出力は使わない。

## 主な実装

- `training/blockwise_bid.py`: 48コマの基準値、上下幅、参加方向を決める入札器
- `market/physical_lp_bidding/`: 物理LP、column generation、Benders
- `market/command_waveforms.py`: 市場の給電目標から指令波形を取り出す
- `training/Agent/`: 分散MADDPG（`maddpg.py` が本研究の学習法、`standard_maddpg.py` が比較用の標準MADDPG）
- `training/system_controller.py`: checkpointの読み込みとforce充電
- `environment/EVEnv.py`: EV群とPCCのBESSを含む環境
- `environment/central_residual_allocator.py`: 中央観測ルールベースの比較器
- `training/evaluate_controller_precision.py`: 入札後の統一評価
- `legacy/finetune/`: 約定後のfine-tune（主線から外した旧実験）
