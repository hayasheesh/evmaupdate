# IEEE TSG形式の日本語LaTeX原稿

`paper.tex`は、IEEE公式のTSG用テンプレートに、`overview/draft_ja_tsg_20261015.docx`の内容を組んだ作業稿である。題目、著者、要旨、本文、表8点、図9点、数式5本、参考文献24件と予定の注記をWordから移している。内容の取り込み元とファイルのSHA-256は`draft_assets/content_manifest.json`に記録している。[コード確認]

## 公式テンプレート

[IEEE公式のTSG向けテンプレート取得画面](https://template-selector.ieee.org/secure/templateSelector/downloadTemplate?publicationTypeId=1&titleId=67&articleId=3&fileId=273)から、2026年10月15日に`IEEE-Transactions-LaTeX2e-templates-and-instructions.zip`を取得した。

`templates/ieee_tsg_official/`に、配布ZIPと展開した全ファイルを保存している。元の`bare_jrnl_new_sample4.tex`、クラス、説明書、サンプルPDFを含む。`source.json`で取得元とハッシュを確認できる。作業用の`IEEEtran.cls`は配布されたクラスとバイト単位で同一である。[コード確認]

`paper.tex`は公式サンプルの`\documentclass[lettersize,journal]{IEEEtran}`とパッケージ設定を使い、日本語用のフォントと表の組版用パッケージを追加している。Letter判、2段組、余白、見出し、図表キャプションの設定はIEEEtranに従う。[コード確認]

## 日本語と英語の文字サイズ

欧文はTimes New Roman、和文は游明朝を使用する。本文は英語10ptに対して日本語を0.92倍の9.2ptとし、見た目の大きさをそろえる。題目、要旨、見出し、表、キャプションもIEEEtranの文字サイズに同じ倍率を適用する。太字の日本語には游明朝Demiboldを使う。[コード確認]

Wordに埋め込まれた図は同じ画像データを使う。表は読みやすい文字サイズを保つため、列の多いものを両段にまたがって配置する。本文の未測定箇所、空欄、書誌の要確認の注記は取り込み元の内容を使用する。[コード確認]

## PDFの作成

プロジェクトのルートで実行する。

```powershell
powershell -ExecutionPolicy Bypass -File docs/paper/build.ps1
```

`docs/paper/paper.pdf`を出力する。Tectonicの既定の実行ファイルは`C:\Users\admin\tools\tectonic\tectonic.exe`。別の場所では`-TectonicPath`で指定する。Git Bashでは`bash docs/paper/build.sh`も使用できる。いずれもLaTeXのPDFだけを作成する。

Wordから取り込み直す場合は、先に次を実行する。この操作は`paper.tex`を生成し直すため、LaTeX側の手編集は別に保存する。

```powershell
python -X utf8 docs/paper/import_draft_docx.py
```

研究内容の基準は監修済みの`paper.docx`に置く。取り込みとPDFの作成は、監修済みWordおよび取り込み元のWordを変更しない。[コード確認]

[IEEE PESの投稿案内](https://ieee-pes.org/publications/authors-kit/preparation-and-submission-of-transactions-papers/)では初回投稿は10ページ以内である。日本語の作業稿と英語原稿ではページ数が異なる。[外部一次資料]
