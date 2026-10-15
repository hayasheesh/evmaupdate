"""Import the current Word draft into the downloaded IEEE journal template."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from zipfile import ZipFile

from lxml import etree

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "overview/draft_ja_tsg_20261015.docx"
ASSETS = HERE / "draft_assets"
NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
}

EQUATIONS = {
    1: r"\max \sum_t (u_t+d_t)-w\sum_t |b_{t+1}-b_t|",
    2: r"u_t\in\{0\}\cup[q,\bar{u}_t],\quad d_t\in\{0\}\cup[q,\bar{d}_t]",
    3: r"b_t-u_t\ge P_t^{\min}+m,\quad b_t+d_t\le P_t^{\max}-m",
    4: r"\forall s\in S:\ \exists y_s,\quad g(x,y_s,\xi_s)\le0",
    5: (r"\begin{gathered}r_t=1-2\tanh(e_t/s)\quad(e_t\le e_0),\\"
        r"\text{その先は直線}\end{gathered}"),
}
EQUATION_SOURCE = {
    1: "max Σ_t (u_t + d_t) − w Σ_t |b_{t+1} − b_t|(1)",
    2: "u_t ∈ {0} ∪ [q, ū_t],  d_t ∈ {0} ∪ [q, d̄_t](2)",
    3: "b_t − u_t ≥ P^min_t + m,  b_t + d_t ≤ P^max_t − m(3)",
    4: "∀ s ∈ S : ∃ y_s,  g(x, y_s, ξ_s) ≤ 0(4)",
    5: "r_t = 1 − 2 tanh(e_t / s)（e_t ≤ e_0）、その先は直線(5)",
}


def escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def prose(text: str) -> str:
    tokens: list[str] = []

    def hold(value: str) -> str:
        tokens.append(value)
        return f"\x01{len(tokens)-1}\x02"

    text = re.sub(
        r"［(\d+)］[〜～]［(\d+)］",
        lambda m: hold(r"\cite{" + ",".join(
            f"ref{n}" for n in range(int(m[1]), int(m[2])+1)) + "}"),
        text,
    )
    text = re.sub(r"［(\d+)］", lambda m: hold(r"\cite{ref" + m[1] + "}"), text)
    text = re.sub(
        r"\bP\^(min|max)_t\b",
        lambda m: hold(r"\(P_t^{" + chr(92) + m[1] + r"}\)"), text,
    )
    text = re.sub(
        r"\b([budey])_([ts0])\b",
        lambda m: hold(r"\(" + m[1] + "_" + m[2] + r"\)"), text,
    )
    text = escape(text)
    return re.sub(r"\x01(\d+)\x02", lambda m: tokens[int(m[1])], text)


def text_of(element) -> str:
    return "".join(element.xpath(".//w:t[not(ancestor::w:del)]/text()", namespaces=NS))


def import_draft() -> None:
    ASSETS.mkdir(exist_ok=True)
    source_bytes = SOURCE.read_bytes()
    with ZipFile(SOURCE) as archive:
        root = etree.fromstring(archive.read("word/document.xml"))
        if root.xpath(".//w:ins|.//w:del", namespaces=NS):
            raise ValueError("Resolve tracked changes explicitly before importing.")
        body = list(root.find("w:body", NS))
        rels = {r.get("Id"): r.get("Target") for r in etree.fromstring(
            archive.read("word/_rels/document.xml.rels"))}
        paragraphs = {i: text_of(e) for i, e in enumerate(body)
                      if etree.QName(e).localname == "p"}
        title, author = paragraphs[0], paragraphs[1]
        abstract_i = next(i for i, t in paragraphs.items() if t.startswith("要旨—"))
        keywords_i = next(i for i, t in paragraphs.items() if t.startswith("索引語—"))
        refs_i = next(i for i, t in paragraphs.items() if t == "参考文献")
        template = (HERE / "templates/ieee_tsg_official/bare_jrnl_new_sample4.tex").read_text()
        preamble = template.split(r"\begin{document}", 1)[0]
        preamble += r"""
% Japanese draft: preserve IEEEtran's page geometry and type sizes.
\usepackage{xeCJK}
\setmainfont{Times New Roman}
\setsansfont{Arial}
\setmonofont{Courier New}
\setCJKmainfont[Scale=0.92,BoldFont={Yu Mincho Demibold}]{Yu Mincho}
\setCJKsansfont[Scale=0.92,BoldFont={Yu Gothic Bold}]{Yu Gothic}
\xeCJKsetcharclass{"2460}{"24FF}{1}
\usepackage{booktabs}
\usepackage{xcolor}
\newlength{\tablecontentwidth}
\newcommand{\draftnote}[1]{{\color{black!65}\small #1\par}}
"""
        chunks = [
            "% Content source: overview/draft_ja_tsg_20261015.docx\n" + preamble,
            r"\begin{document}",
            r"\title{" + prose(title) + "}",
            r"\author{" + prose(author) + "}",
            r"\markboth{IEEE Transactions on Smart Grid}{Hayashi \MakeLowercase{\textit{et al.}}: EV群の調整力供出}",
            r"\maketitle",
            r"\begin{abstract}" + prose(paragraphs[abstract_i].split("—", 1)[1]) + r"\end{abstract}",
            r"\begin{IEEEkeywords}" + prose(paragraphs[keywords_i].split("—", 1)[1]) + r"\end{IEEEkeywords}",
        ]
        covered = {0, 1, abstract_i, keywords_i}
        manifest = []
        table_number = 0
        figure_number = 0
        skip = set()
        for i, element in enumerate(body):
            tag = etree.QName(element).localname
            if i in covered or i in skip or i > refs_i:
                continue
            if tag == "tbl":
                continue  # Handled with its caption.
            if tag != "p":
                continue
            text = paragraphs[i]
            blips = element.findall(".//a:blip", NS)
            if blips:
                if len(blips) != 1:
                    raise ValueError(f"Unexpected composite image paragraph {i}")
                figure_number += 1
                member = "word/" + rels[blips[0].get("{" + NS["r"] + "}embed")]
                name = f"figure_{figure_number:02d}.png"
                data = archive.read(member)
                (ASSETS / name).write_bytes(data)
                caption_i = i+1
                caption = re.sub(r"^Fig\.\s*\d+\.\s*", "", paragraphs[caption_i])
                extent = element.find(".//wp:extent", NS)
                wide = int(extent.get("cx")) > 5*914400
                env = "figure*" if wide else "figure"
                chunks += [
                    r"\begin{" + env + r"}[!t]",
                    r"\centering",
                    r"\includegraphics[width=\linewidth]{draft_assets/" + name + "}",
                    r"\caption{" + prose(caption) + "}",
                    r"\label{fig:" + str(figure_number) + "}",
                    r"\end{" + env + "}",
                ]
                covered.update([i, caption_i])
                skip.add(caption_i)
                manifest.append({"kind": "figure", "source_blocks": [i, caption_i],
                                 "caption": caption, "source_media": member,
                                 "file": name, "sha256": hashlib.sha256(data).hexdigest()})
                continue
            if not text:
                continue
            if re.fullmatch(r"TABLE [IVX]+", text):
                table_number += 1
                caption_i, table_i = i+1, i+2
                table = body[table_i]
                assert etree.QName(table).localname == "tbl"
                rows = [
                    [" ".join(text_of(p) for p in cell.findall("w:p", NS))
                     for cell in row.findall("w:tc", NS)]
                    for row in table.findall("w:tr", NS)
                ]
                ncols = len(rows[0])
                widths = [int(cell.find("w:tcPr/w:tcW", NS).get("{" + NS["w"] + "}w"))
                          for cell in table.find("w:tr", NS).findall("w:tc", NS)]
                wide = table_number in {1, 3, 4, 6, 8}
                env = "table*" if wide else "table"
                spec = "@{}" + "".join(
                    r">{\raggedright\arraybackslash}p{" +
                    f"{w/sum(widths):.7f}" + r"\tablecontentwidth}"
                    for w in widths) + "@{}"
                chunks += [
                    r"\begin{" + env + r"}[!t]",
                    r"\caption{" + prose(paragraphs[caption_i]) + "}",
                    r"\label{tab:" + str(table_number) + "}",
                    r"\centering\footnotesize",
                    r"\setlength{\tablecontentwidth}{\dimexpr\linewidth-" +
                    str(2*(ncols-1)) + r"\tabcolsep\relax}",
                    r"\begin{tabular}{" + spec + "}",
                    r"\toprule",
                    " & ".join(prose(cell) for cell in rows[0]) + r" \\",
                    r"\midrule",
                ]
                for row in rows[1:]:
                    chunks.append(" & ".join(prose(cell) for cell in row) + r" \\")
                chunks += [r"\bottomrule", r"\end{tabular}", r"\end{" + env + "}"]
                covered.update([i, caption_i, table_i])
                skip.update([caption_i, table_i])
                manifest.append({"kind": "table", "source_blocks": [i, caption_i, table_i],
                                 "caption": paragraphs[caption_i], "rows": rows})
                continue
            if i == refs_i:
                chunks.append(r"\begin{thebibliography}{24}")
                covered.add(i)
                continue
            if text == ". はじめに":
                chunks.append(r"\section{はじめに}")
                kind = "section"
            elif re.match(r"^[IVX]+\.\s+", text):
                chunks.append(r"\section{" + prose(text.split(". ", 1)[1]) + "}")
                kind = "section"
            elif re.match(r"^[A-Z]\.\s+", text):
                chunks.append(r"\subsection{" + prose(text.split(". ", 1)[1]) + "}")
                kind = "subsection"
            elif re.search(r"\([1-5]\)$", text) and re.match(
                    r"^(max Σ_t|u_t ∈|b_t −|∀ s ∈ S|r_t =)", text):
                number = int(re.search(r"\((\d)\)$", text)[1])
                if text != EQUATION_SOURCE[number]:
                    raise ValueError(f"Review the changed equation {number} explicitly.")
                chunks.append(r"\begin{equation}" + EQUATIONS[number] +
                              r"\label{eq:" + str(number) + r"}\end{equation}")
                kind = "equation"
            elif text.startswith("【予定】"):
                chunks.append(r"\draftnote{" + prose(text) + "}")
                kind = "note"
            else:
                chunks.append(prose(text) + "\n")
                kind = "paragraph"
            covered.add(i)
            manifest.append({"kind": kind, "source_blocks": [i], "text": text})
        references = []
        for i in range(refs_i+1, len(body)):
            text = paragraphs.get(i, "")
            if not text:
                continue
            match = re.fullmatch(r"\[(\d+)\] (.*)", text)
            if not match:
                raise ValueError(f"Unexpected reference paragraph: {text}")
            chunks.append(r"\bibitem{ref" + match[1] + "} " + prose(match[2]))
            references.append(text)
            covered.add(i)
        chunks += [r"\end{thebibliography}", r"\end{document}", ""]
        missing = [i for i, e in enumerate(body)
                   if (paragraphs.get(i) or etree.QName(e).localname == "tbl"
                       or e.findall(".//a:blip", NS)) and i not in covered]
        if missing:
            raise ValueError(f"Unconverted blocks: {missing}")
        (HERE / "paper.tex").write_text("\n".join(chunks), encoding="utf-8")
        result = {
            "source": SOURCE.relative_to(HERE).as_posix(),
            "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "title": title, "authors": author,
            "abstract": paragraphs[abstract_i], "keywords": paragraphs[keywords_i],
            "converted_nonempty_blocks": len(covered),
            "unconverted_blocks": missing, "tables": table_number,
            "figures": figure_number, "references": references, "blocks": manifest,
        }
        (ASSETS / "content_manifest.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({k: result[k] for k in [
            "source", "source_sha256", "converted_nonempty_blocks",
            "unconverted_blocks", "tables", "figures"]}, ensure_ascii=False))


if __name__ == "__main__":
    import_draft()
