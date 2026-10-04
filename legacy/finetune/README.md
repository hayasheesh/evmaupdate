# Fine-tune旧実験

このフォルダは、pretrain済み方策を未見日の入札へ追加学習させた旧実験を保存する。
現在の提案システムはfine-tuneを使用しない。

- 実装: `fine_tune.py`
- 条件と結果: `RESULTS.md`

再現する場合はプロジェクトルートから次を実行する。

```powershell
python legacy/finetune/fine_tune.py --day 2024-12-04 --warmstart <pretrain run>
```

共有学習器には旧実験との互換性を保つためfine-tune分岐が残っている。通常の
`pre_train.py`および最終評価からは呼ばれない。
