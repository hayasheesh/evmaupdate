# 研究室PC の 256指令入札まわりの成果（写し）

2026-10-02 22:40 に研究室PC（smartgrid-gpu、/home/hayashi/workspace/EVMALOCALUPDATE）から写した。研究室と同じフォルダ構成で置いてある。
入札の設計は文書（docs/入札器の問題設定.md）どおり「設計指令128本 × EV 3本」に統一したため、ここにある 256本 × EV 3本の実行は本線ではない。

## 入っているもの

- archive/prod_aemoplan_AB_20station_5080_scaledreward_20261001_234702
  20 station、AEMO、256本の入札での AB 学習。大域報酬の kW を 20/7 倍（428.6 / 171.4 kW）。900回で止めた（途中テストの和は 900回目で 178.6）。
  グラフ・CSV・テスト記録（results/）、TensorBoard の記録（performance/、runs/）、コードの写し、最後の 900回の行動器（results/TEST900/actor_*）。
- archive/prod_aemoplan_AB_20station_5080_20261001_222317
  同じ条件で、コンテナが GPU を失っていたため CPU だけで5回まで進んだ実行。報酬の kW は 7 station と同じ 150 / 60 kW。
- execute_results/bid_banks/train_25_20station_256cmd_3ev_aemo_plan_deviation、validation_5_20station_256cmd_3ev_aemo_plan_deviation
  上の学習が使った入札バンク（学習用25日、検証用5日）。
- execute_results/bid_banks/train_25_20station_256cmd_3ev_ercot_plan_deviation
  ERCOT の 256本の入札の作りかけ（学習用 8日分。途中で止めた）。
- execute_results/remote_20station_bid_unseen_20261002
  上の AEMO 256本の学習用入札の、未見指令での物理的成立の判定（25日 × 256本 × 独立EV 3本）。成立 17,214、不成立 1,810、時間切れ 176（89.7〜90.6%）。
- execute_results/remote_5080_20station_*
  研究室で動かした起動スクリプトとログ。

## 外したもの（研究室には残っている）

- 学習の完全な状態（resume/training_state_ep700/800/900.pth、合わせて約 6 GB）
- エージェント全体の保存（results/TEST*/agent_state_*、約 1.8 GB）と経験の保存（replay_snapshot_*、約 240 MB）
- 900回以外の途中テストの行動器（results/TEST20〜TEST880/actor_*）
