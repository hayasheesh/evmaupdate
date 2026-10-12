# 研究室の現行20局学習と監視

2026-10-08 02:27 JSTにログと実行プロセスを確認した。

## 接続と対象

Windowsから `C:\Windows\System32\OpenSSH\ssh.exe -o BatchMode=yes -o ConnectTimeout=15 smartgrid-gpu` で接続できる。接続先はUbuntuのhayashiユーザー。ホストの `/home/hayashi/workspace` はコンテナ `hayashi-gpu` の `/workspace` に対応する。Pythonは `/workspace/.venvs/evma20/bin/python`。秘密鍵は開かずコピーしない。

本体はホスト `/home/hayashi/workspace/EVMALOCALUPDATE`。現行2本は次の実行である。

| 市場 | archiveの実行名 | 状態とログ |
|---|---|---|
| AEMO | allin_aemoplan_AB_20station_20261007_042318 | execute_results/allin_20station_aemo_20261007 |
| ERCOT | allin_ercotplan_AB_20station_20261007_042436 | execute_results/allin_20station_ercot_20261007 |

02:27にAEMO 1082回、ERCOT 1056回、両方の親と子が稼働中だった。エラーログにTracebackはなく、GPUは53℃、使用メモリ11,142 MiB、空きRAMは23.03 GiBだった。現行状態はAEMO1634回、ERCOT1595回で完全状態を保存して安全停止。20局は原因分析の対象で、再開しない。

## 現在の設定

EV到着3倍、候補128通りから下側2通り＋総台数最多を選び、設計指令128本との384組で入札を作る。最低入札量800 kW、アセスメントIの余裕80 kW、基準値の段差の重み0.5。観測に休日区分と帯幅を加えている。追従報酬はconvex。差分報酬と残差のstation数割りは無効。Q_MIX_GLOBAL_WEIGHTは0.5、リプレイバッファは350,000遷移。

Windows7局はAEMO ep812、ERCOT ep928の完全状態から再開し、目標は両方2000回。実行はarchive/allin_aemoplan_AB_7station_20261007_165254とallin_ercotplan_AB_7station_20261007_155605。ログはexecute_results/allin_7station_20261007/logs/{aemo,ercot}_resume_codex_20261008.{stdout,stderr}.log。特徴比較の再学習は中止し、保存済みの固定入力比較は不要候補の根拠として扱う。再学習での不要性は未測定。Windows boostはAC/DCとも0。

## 確認と結果の写し

Windowsで次を実行すると、研究室の既存状態を読み、現行2本の図表とTensorBoard記録をWindowsのarchiveへ写す。

```powershell
& 'C:\Users\admin\AppData\Local\Programs\Python\Python310\python.exe' 'C:\Users\admin\Desktop\EVMALOCALUPDATE\execute_results\watch_codex_20261008\check_lab.py' --mirror
```

チェックポイント、行動器、リプレイバッファは写さない。学習の完全状態は研究室側にある。既存実行のコードと入札は変更しない。再開時にはソースと設定の照合がある。

成績CSVは途中テストのエピソード平均で、ログのsummaryはstep・EVを集計した値である。旧health.pyの追従率再構成はCSVと一致しないため、監視の成績は現在のCSVから直接読む。混合して比較しない。勾配の診断には既存TensorBoardを読む。gc_clipの大きさだけで発散と断定しない。

02:27時点のCSVの最新成績は、AEMO TEST1080で追従78.29%、SoC86.32%、和164.61。ERCOT TEST1040で84.55%、77.74%、和162.29。AEMOは直近の和が164.4〜165.2、ERCOTは160.1〜162.3だった。

監視は30分ごと。進行が変わらない間は通知しない。完了、停止、例外、30分以上の真の停滞、持続する成績悪化、1500/2000回到達、GPU異常、メモリ不足、接続失敗が続く場合に通知する。1000回到達は確認済み。前回状態と通知済み事項はmonitor_state.jsonに残す。

重大なGPU異常で安全停止が必要な場合は、利用者が許可した `pre_train.py --request-stop <コンテナ内run_dir>` で現在回の保存後に止める。OS、Docker、ドライバ、電源設定を変更せず、旧実行を削除・移動しない。新しい実験・学習・入札は起動しない。

## 研究の引き継ぎ

Claudeの10/5〜10/8の5セッションを重複を除いて確認した。ソースは `C:\Users\admin\.claude\projects\C--Users-admin-Desktop-EVMALOCALUPDATE\` の891fc5c2、60458f74、aaa9d98a、86413059、10178949で始まるJSONL。

最新の研究方針はdocs/研究の全体像.mdにある。平均の追従性能と市場の多段階の判定の差を測り、帯を外した時刻を3つの反実仮想の分類で調べ、制御器の誤差を入札へ戻す。出発時SoCを含めて状態管理とその時刻の協調を分けるLP、および誤差分布を入札へ反映する実験は未完である。

P90は追従率でなく供出可能量の可用性として扱う。過去の5分刻みの集計を、実市場の秒・1分値の正式な適合結果とは言わない。BESSは必要最小量を測り、EVとBESSの調整力の割合を最終評価で出す。複数地点に置ける前提で合計2 MWを上限にしない。20局の最低入札量を1 MWへ変える案は承認されておらず800 kWを維持する。コマ内残差累計の観測は保留。

過去の比較値や先行研究の主張を現行学習の結果として扱わない。新規性は確認した文献の範囲と未測定の実験を分けて報告する。文書には現在の設計と確認した事実を記し、会話だけの経緯を設計の説明に混ぜない。監修済みの文書は編集しない。
