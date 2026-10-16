"""draft_ja.txt（日本語の本文）を IEEE TSG の体裁に組んだ Word を作る（2026-10-15）。

    python make_figs.py && python make_draft_docx.py      draft_ja_tsg_20261015.docx を書く

draft_ja.txt の書き方
    # 題                    題目
    @abstract の次の行      要旨（1段落）
    @index 語、語           索引語
    ## 節 / ### 小節        見出し
    %% 文                   まだ測っていないことや未決のことの注記（灰色）
    @table 名前             下の TABLES の表
    @fig 名前 | 説明         1段幅の図、@widefig は2段抜きの図
    @eq 式                  式（番号を付ける）
    @references の後の行    参考文献（1行1本、上から番号）
    それ以外の行            本文の段落
監修済みの docs/paper/paper.docx とは別のファイル。

利用者が Word で赤字で書いた段落（要旨と索引語）は、作り直すと消える。作り直したら、前の Word から赤字の段落を
移す（10/16 の組み直しでは moved_aside の前の版から移した）。IEEE の体裁にするのは format_draft_ieee_word.py。
"""
from __future__ import annotations

from pathlib import Path

from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

from make_overview_docx import NOTE, Paper

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'draft_ja.txt'
OUT = HERE / 'draft_ja_tsg_20261015.docx'

TABLES = {
    'rules': ('追従の判定：市場ごとの比較［15］［16］［22］［23］', ['市場', '判定の単位', '許容', '続けて参加する条件', '外れたとき'], [
        ['日本 二次②', '1分値・30分コマ', '約定量の±10%', '30点中27点。月3回の不適合で停止のおそれ', 'コマの料金を返す'],
        ['PJM', '10秒・5分の点数', '誤差÷(0.5|信号|＋0.5 AReg)', '100時間の平均40%以上', '点数に比例。25%未満は0'],
        ['CAISO', '4秒・15分', '偏差の和÷設定値の和', '月平均25%以上', '支払いに精度を掛ける'],
        ['ERCOT', '5分平均', 'max(X%, Y MW)', '月の区間の85%', '報告'],
        ['ドイツ aFRR', '1秒', '許容の幅＋5%', '5%を超えた不足を計上', '不足量×価格'],
        ['英国 DC', '20 Hz・30分', '3%で減額、7%で0', '期間内の最悪値', '支払いの減額'],
    ], [0.55, 0.62, 0.78, 0.85, 0.65], 6.5),
    'env': ('環境と評価の条件', ['項目', '設定'], [
        ['station', '6（職場3・住宅3）と20（職場10・住宅10）'],
        ['EV', 'ACN（JPL、Caltech）とノルウェーの住宅の実セッション。定格7.7 kW'],
        ['指令', 'AEMO・ERCOT（計画からのずれ）、PJM RegD'],
        ['規則', '日本の二次②（アセスメントI・II）。下げも約定できると仮定'],
        ['入札', '前日、30分コマ48、設計用の指令128本、EVの実現3通り'],
        ['評価', '検証用の平日5日 × 未見の指令4本 × EVの乱数2通り＝40組'],
        ['学習', 'CTDE、2000回。行動器は自局・入札・指令・1つ前の全体のずれ'],
    ], [0.6, 2.75], 7.0),
    'prior': ('先行研究の方式の成績（6 station、40組）', ['方式', '市場', '5分', 'コマ/日', 'SoC（平均/最小）', '入札量'], [
        ['Liu/Lyu の入札と配分', 'AEMO', '99.5%', '1.02', '89.0/37.5%', '20,095'],
        ['Liu/Lyu の入札と配分', 'ERCOT', '99.3%', '0.60', '88.9/58.4%', '12,681'],
        ['Liu/Lyu の入札と配分', 'PJM', '99.1%', '0.30', '69.9/39.0%', '1,198'],
        ['本研究の入札＋Liu/Lyu の配分', 'AEMO', '99.1%', '0.97', '93.3/46.9%', '15,749'],
        ['本研究の入札＋Liu/Lyu の配分', 'ERCOT', '98.7%', '1.43', '93.1/66.3%', '16,220'],
        ['本研究の入札＋Liu/Lyu の配分', 'PJM', '87.7%', '24.3', '64.7/32.4%', '16,783'],
        ['MARL-AFM［12］', 'AEMO・ERCOT', '未測定', '未測定', '未測定', '未測定'],
    ], [1.0, 0.5, 0.44, 0.4, 0.64, 0.46], 6.5),
    'first': ('最初のやり方の成績（7 stationの旧環境の参考値。B0 の結果で置き換える）', ['市場', '方法', '1日すべて', '5分', 'SoC'], [
        ['AEMO', '全情報のLP', '73.3〜74.7%', '—', '—'], ['AEMO', '中央のルールベース', '47.5%', '96.6%', '100%'],
        ['AEMO', '分散MARL', '0.3%', '90.5%', '82.8%'], ['ERCOT', '全情報のLP', '80.8〜81.7%', '—', '—'],
        ['ERCOT', '分散MARL', '0%', '88.7%', '65.8%'], ['AEMO・ERCOT', 'やり方1（今の環境）', '未測定', '未測定', '未測定'],
    ], [0.65, 1.1, 0.65, 0.45, 0.45], 6.5),
    'adapt': ('手当てとEVの選び方の有無（6 station、40組）。入札は市場と手当ての有無、入札量はkW・コマ/日、LPは全情報のLPで成り立つ組、ルールは中央のルールベースの30分コマの合格率。追従（5分ごと）、SoC、コマ（30分コマの合格率）は分散MARL',
              ['入札', '入札量', 'LP', 'ルール', '追従', 'SoC', 'コマ'], [
        ['AEMO あり', '15,749', '39/40', '98.1%', '95.0%', '87.6%', '89.8%'],
        ['AEMO なし', '20,124', '35/40', '96.6%', '94.7%', '82.5%', '90.1%'],
        ['ERCOT あり', '16,220', '40/40', '98.1%', '91.2%', '83.6%', '87.2%'],
        ['ERCOT なし', '20,455', '36/40', '96.8%', '93.6%', '78.7%', '89.8%'],
    ], [0.75, 0.45, 0.4, 0.42, 0.42, 0.42, 0.42], 6.5),
    'selection': ('標本の取り方と未見通過率（6 station、1,920組）。各欄は入札量（kW・コマ/日）/ 未見通過率。予測精度の行は未見通過率だけ', ['変種', 'AEMO', 'ERCOT'], [
        ['本線（指令128本、EVは下2上1）', '15,749 / 99.27%', '16,220 / 99.79%'],
        ['指令8本', '15,433 / 99.17%', '16,634 / 99.64%'],
        ['指令16〜64本', '16,089〜16,807 / 99.06%', '16,774〜16,786 / 99.64%'],
        ['指令256本', '15,655 / 99.22%', '16,210 / 99.79%'],
        ['EV 1通り', '17,804 / 91.98%', '18,218 / 94.43%'],
        ['EV 2通り', '17,658 / 88.70%', '18,119 / 95.31%'],
        ['EV 4通り', '15,184 / 99.48%', '15,571 / 100%'],
        ['EV 6通り', '14,523 / 99.48%', '15,145 / 100%'],
        ['下1・中央・最大', '17,114 / 91.67%', '17,647 / 98.07%'],
        ['最小・中央・最大', '18,401 / 94.17%', '18,825 / 96.72%'],
        ['無作為3通り（3回）', '18,312〜20,323 / 92.1〜94.2%', '18,814〜20,799 / 95.2〜96.6%'],
        ['予測精度95・91・85%（下2上1）', '99.3・96.0・87.3%', '99.7・97.3・91.5%'],
        ['予測精度95・91・85%（無作為）', '98.4・84.0・66.8%', '99.0・89.3・75.8%'],
    ], [1.25, 1.1, 1.1], 6.5),
    'market': ('市場の要求との照合（40組。不適合は月20日に換算）', ['規模', '市場', '方法', '30分コマ', '月の不適合', '5分'], [
        ['6', 'AEMO', '分散MARL', '89.8%', '98', '95.0%'], ['6', 'AEMO', 'ルールベース', '98.1%', '18', '98.9%'],
        ['6', 'ERCOT', '分散MARL', '87.2%', '123', '91.2%'], ['6', 'PJM', '分散MARL', '74.9%', '241', '89.5%'],
        ['20', 'AEMO', '分散MARL', '90.7%', '90', '95.9%'], ['20', 'ERCOT', '分散MARL', '88.6%', '109', '92.5%'],
        ['20', 'PJM', 'ルールベース', '93.1%', '66', '98.3%'], ['20', 'PJM', '分散MARL', '学習中', '学習中', '学習中'],
    ], [0.35, 0.5, 0.8, 0.5, 0.6, 0.5], 6.5),
    'latency': ('処理時間', ['処理', '規模', '値'], [
        ['制御の1ステップ（行動器・強制充電・蓄電池）', '6 station', '95%点 13〜16 ms、最大 80 ms'],
        ['制御の1ステップ', '20 station', '未測定'], ['入札の作成（1日）', '6・20 station', '未測定'],
    ], [1.6, 0.75, 1.0], 6.5),
    'pjm': ('日本の判定とPJMの判定（6 station PJM、40組）', ['方法', '日本の30分コマ', 'PJM 5分の点数', '支払い0の5分', '100時間平均の最小'], [
        ['分散MARL', '74.9%', '94.5%', '0.03%', '92.3%'], ['行動器＋強制充電', '57.9%', '88.1%', '1.5%', '83.9%'],
        ['中央のルールベース', '95.1%', '99.4%', '0.18%', '98.1%'],
        ['PJMの規則で入札・学習（分散MARL）', '—', '93.6%', '0.10%', '91.3%'],
        ['PJMの規則の入札（中央のルールベース）', '—', '96.4%', '1.95%', '93.0%'],
    ], [1.05, 0.55, 0.65, 0.55, 0.6], 6.5),
}


def build():
    P = Paper()
    P.doc.styles['Normal'].paragraph_format.line_spacing = Pt(12)
    lines = SOURCE.read_text(encoding='utf-8').splitlines()
    i = 0
    in_refs = False
    ref = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        if in_refs:
            ref += 1
            p = P.para(f'[{ref}] {line}', size=8, indent=False)
            p.paragraph_format.left_indent = Inches(0.25)
            p.paragraph_format.first_line_indent = Inches(-0.25)
            continue
        if line.startswith('# '):
            P.para(line[2:], size=18, indent=False, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)
            P.para('（日本語の下書き。灰色の注記は、まだ測っていないことや決めていないこと）', size=8, color=NOTE,
                   indent=False, align=WD_ALIGN_PARAGRAPH.CENTER)
            P.para('Koshin Hayashi, Sihui Xue, and Daisuke Kodaira', size=11, indent=False, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=8)
            P.new_section(2)
        elif line == '@abstract':
            text = lines[i].strip()
            i += 1
            p = P.para('', indent=False)
            r = p.add_run('要旨—')
            r.bold = True
            p.add_run(text)
        elif line.startswith('@index '):
            p = P.para('', indent=False, space_after=4)
            r = p.add_run('索引語—' + line[len('@index '):])
            r.bold = True
        elif line.startswith('### '):
            P.subsection(line[4:])
        elif line.startswith('## '):
            P.section(line[3:])
        elif line.startswith('%% '):
            P.note(line[3:])
        elif line.startswith('@table '):
            caption, header, rows, widths, size = TABLES[line[len('@table '):]]
            P.table(caption, header, rows, widths, size)
        elif line.startswith('@widefig '):
            name, caption = (x.strip() for x in line[len('@widefig '):].split('|', 1))
            P.wide_figure(name, caption)
        elif line.startswith('@fig '):
            name, caption = (x.strip() for x in line[len('@fig '):].split('|', 1))
            P.figure(name, caption)
        elif line.startswith('@eq '):
            P.equation(line[len('@eq '):])
        elif line == '@references':
            P.para('参考文献', size=10, indent=False, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
            in_refs = True
        else:
            P.para(line)
    P.doc.save(OUT)
    chars = sum(len(l) for l in lines if l.strip() and not l.startswith(('@', '#', '%%')))
    print('body characters', chars, 'figures', P.fig, 'tables', P.tab, 'references', ref, '->', OUT)


if __name__ == '__main__':
    build()
