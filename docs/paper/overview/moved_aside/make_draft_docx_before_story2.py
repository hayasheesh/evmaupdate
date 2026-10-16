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

draft_ja_tsg_20261015.docx は、題目・要旨・索引語を利用者が Word で直し、その後の数値も Word に直接書き込んでいる。
このスクリプトで作り直すとそれらが消えるので、作り直さない。IEEE の体裁にするのは format_draft_ieee_word.py。
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
        ['本研究の入札＋Liu/Lyu の配分', 'AEMO', '99.1%', '0.97', '93.3/46.9%', '15,749'],
        ['本研究の入札＋Liu/Lyu の配分', 'ERCOT', '98.7%', '1.43', '93.1/66.3%', '16,220'],
        ['MARL-AFM［12］', 'AEMO・ERCOT', '未測定', '未測定', '未測定', '未測定'],
    ], [1.0, 0.5, 0.44, 0.4, 0.64, 0.46], 6.5),
    'first': ('最初のやり方の成績（7 stationの旧環境の参考値。B0 の結果で置き換える）', ['市場', '方法', '1日すべて', '5分', 'SoC'], [
        ['AEMO', '全情報のLP', '73.3〜74.7%', '—', '—'], ['AEMO', '中央のルールベース', '47.5%', '96.6%', '100%'],
        ['AEMO', '分散MARL', '0.3%', '90.5%', '82.8%'], ['ERCOT', '全情報のLP', '80.8〜81.7%', '—', '—'],
        ['ERCOT', '分散MARL', '0%', '88.7%', '65.8%'], ['AEMO・ERCOT', 'やり方1（今の環境）', '未測定', '未測定', '未測定'],
    ], [0.65, 1.1, 0.65, 0.45, 0.45], 6.5),
    'selection': ('EVの選び方と設計用の指令の本数（6 station ERCOT、1,920組）', ['変種', '入札量（kW・コマ/日）', '未見通過率'], [
        ['本線（指令128本、下2上1）', '16,220', '99.79%'], ['指令8・16・32・64本', '16,634〜16,786', '99.64%'],
        ['EV 1通り / 2通り', '18,218 / 18,119', '94.43% / 95.31%'], ['EV 4通り / 6通り', '15,571 / 15,145', '100% / 100%'],
        ['下1・中央・最大', '17,647', '98.07%'], ['無作為3通り（3回）', '18,814〜20,799', '95.16〜96.61%'],
        ['予測精度95・91・85%', '測定中', '測定中'], ['AEMO の全部', '測定中', '測定中'],
    ], [1.55, 1.0, 0.8], 6.5),
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
        ['中央のルールベース', '95.1%', '99.4%', '0.18%', '98.1%'], ['PJMの規則で入札・学習', '—', '学習待ち', '学習待ち', '学習待ち'],
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
