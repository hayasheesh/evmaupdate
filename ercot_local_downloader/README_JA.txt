ERCOT SCED 指令波形の取得
========================

目的：波形だけを抽出。EV-VPPの基準値・入札上下幅はこちらで決める。

取得するのは2025-12-05以降のNP3-965-ER。
標準では全Load Resource（CLR候補）と全ESR。Generationは任意追加。
旧resources.txtのAR_ALD1/BOERNE_ALD1制限は新しいCMDでは使わない。

ユーザーが実行する入口
----------------------
1. run_test_windows.bat：1日だけ取得し、接続とCSV列を確認する。
2. run_all_windows.bat：全期間取得する。
3. run_cop_windows.bat：COP調整期間スナップショット（NP1-301）を全期間取得する。
4. run_plan_probe_windows.bat：1営業日分のDAM・COP・SCEDを全列で取得し、列を確認する。

既存のPublic API認証を再利用し、email、password、subscription keyを入力する。
passwordとkeyは非表示入力。認証情報は出力ファイルに保存しない。
bank生成や学習は自動実行しない。

全取得の保存先：
  ercot_sced_rtc_output/manifest.json
  ercot_sced_rtc_output/documents/*.csv
テストの保存先：
  ercot_sced_rtc_test_output/

既定の終了日は実行日の60日前。再開時はmanifestの範囲を維持する。
同じCMDで再開可能。完了済みCSVのハッシュを確認し、未完了文書だけ処理する。
期間を変えて新規取得する場合：
  python download_ercot_sced_waveforms.py --end YYYY-MM-DD --out-dir 新しいフォルダ

任意のGeneration追加：
  python download_ercot_sced_waveforms.py --include-generation --out-dir 別のフォルダ

認証・通信なしで計画だけ確認：
  python download_ercot_sced_waveforms.py --dry-run

取得方式
--------
ESRの未確認の行APIを推測せず、公式アーカイブAPIからNP3-965-ERのZIPを取得する。
旧CLR専用行APIより通信量は増える。巨大ZIPは保存せず、必要CSVの必要列だけ残す。
原SCED時刻、resource種別、status、Base Point、MPC/LPCまたはHSL/LSL、公開日・文書URLを保持する。
後日訂正も対象にし、整形時に公開日が新しい行を優先する。
CSVの欠損をこの段階で補間しない。

次の整形・監査手順はプロジェクトの
  docs/指令データ.md
を参照。期間境界は監査後に明示する。合成シナリオ数を実日数と混同しない。

旧download_ercot_ader_sced_api.py、旧resources.txt、過去の取得成果物は残してある。
従来の名前指定CLR専用取得を再現する場合だけ旧スクリプトを直接指定する。
