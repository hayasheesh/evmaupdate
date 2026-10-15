"""Format the saved Word draft using the official IEEE TSG Word package.

    python format_draft_ieee_word.py      draft_ja_tsg_20261015.docx -> draft_ja_tsg_20261015_IEEE.docx

The source is the saved Word file, not draft_ja.txt: the author edited its title,
abstract and keywords in Word. Body text is copied unchanged, except that the
first heading gets back its numeral "I" (lost in that edit). Latin text is
10 pt and Japanese 9 pt, on the template's 12 pt line pitch.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from lxml import etree

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "draft_ja_tsg_20261015.docx"
REFERENCE = HERE.parent / "templates/ieee_tsg_word_official/Transactions-template.docx"
OUTPUT = HERE / "draft_ja_tsg_20261015_IEEE.docx"
QA = HERE.parents[2] / "tmp/docs/tsg_word"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
PKG = "http://schemas.openxmlformats.org/package/2006/relationships"
NS = {"w": W, "r": R, "a": A}


def q(name):
    return "{" + W + "}" + name


def sub(parent, name, **attrs):
    child = parent.find(q(name))
    if child is None:
        child = etree.SubElement(parent, q(name))
    for key, value in attrs.items():
        child.set(q(key), str(value))
    return child


def xml(data):
    return etree.fromstring(data)


def encoded(root):
    return etree.tostring(root, encoding="UTF-8", xml_declaration=True, standalone=True)


def text(element):
    return "".join(element.xpath(".//w:t/text()", namespaces=NS))


def package(path):
    with ZipFile(path) as archive:
        return {n: archive.read(n) for n in archive.namelist()}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def is_japanese(character):
    code = ord(character)
    return (0x2E80 <= code <= 0x9FFF or 0xF900 <= code <= 0xFAFF
            or 0xFF01 <= code <= 0xFF60 or 0x20000 <= code <= 0x2FFFF
            or 0x2460 <= code <= 0x24FF)


def split_and_format_runs(paragraph, size, bold=None, italic=None):
    for run in list(paragraph.findall(".//" + q("r"))):
        if any(etree.QName(e).localname not in {"rPr", "t", "tab", "br", "cr"}
               for e in run):
            continue
        parent = run.getparent()
        base = copy.deepcopy(run.find(q("rPr")))
        if base is None:
            base = etree.Element(q("rPr"))
        pieces = []
        for node in run:
            if node.tag == q("t"):
                value = node.text or ""
                start = 0
                for end in range(1, len(value)+1):
                    if end == len(value) or is_japanese(value[end]) != is_japanese(value[start]):
                        pieces.append((value[start:end], is_japanese(value[start])))
                        start = end
            elif node.tag in {q("tab"), q("br"), q("cr")}:
                pieces.append((copy.deepcopy(node), False))
        if not pieces:
            continue
        index = parent.index(run)
        for offset, (piece, japanese) in enumerate(pieces):
            replacement = etree.Element(q("r"))
            props = copy.deepcopy(base)
            for name in ("sz", "szCs", "rFonts", "lang", "w", "position", "spacing",
                         "rStyle", "fitText", "szScale"):
                for old in props.findall(q(name)):
                    props.remove(old)
            sub(props, "rFonts", ascii="Times New Roman", hAnsi="Times New Roman",
                eastAsia="Yu Mincho", cs="Times New Roman")
            points = round(size*.92*2)/2 if japanese else size
            sub(props, "sz", val=round(points*2))
            sub(props, "szCs", val=round(points*2))
            sub(props, "lang", val="ja-JP" if japanese else "en-US", eastAsia="ja-JP")
            for name, value in (("b", bold), ("i", italic)):
                if value is not None:
                    sub(props, name, val="1" if value else "0")
            replacement.append(props)
            if isinstance(piece, str):
                t = etree.SubElement(replacement, q("t"))
                t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
                t.text = piece
            else:
                replacement.append(piece)
            parent.insert(index+offset, replacement)
        parent.remove(run)


# Exact line pitch by font size (twips). Yu Mincho under "auto" spacing takes
# about 15 pt per 9 pt line, against 12 pt for the template's 10 pt Times.
EXACT_LINE = {10: 240, 9: 220, 8: 200}
COLUMN_TWIPS = 5040  # 3.5 in, one column of the template's two-column section
CELL_PADDING_TWIPS = 216
UNIT_TWIPS = 150  # one 7.5 pt Japanese character in a table cell


def display_units(value):
    # Widths relative to a 7.5 pt Japanese character, for 8 pt Times.
    return sum(1.0 if is_japanese(c) else 0.9 if c == '%' else 0.7 if c.isupper()
               else 0.35 if c in ".,:;()/-'" else 0.55 for c in value)


def fit_table_widths(table):
    """Share one column's width among table columns by their text length.

    Latin words (ERCOT, 16,220, 89.0/37.5%) cannot wrap, so each column is at least as wide
    as its longest Latin word; the rest goes to columns with longer text.
    """
    rows = table.findall(q("tr"))
    count = max(len(r.findall(q("tc"))) for r in rows)
    want, least = [0.0] * count, [2.0] * count
    for row in rows:
        for j, cell in enumerate(row.findall(q("tc"))):
            value = text(cell)
            want[j] = max(want[j], display_units(value))
            for word in re.split(r"[\s、，／・（）()]+", value):
                if word and not any(is_japanese(c) for c in word):
                    least[j] = max(least[j], display_units(word))
    want_tw = [u * UNIT_TWIPS + CELL_PADDING_TWIPS for u in want]
    least_tw = [u * UNIT_TWIPS * 1.08 + CELL_PADDING_TWIPS for u in least]
    if sum(want_tw) <= COLUMN_TWIPS:
        widths = [w * COLUMN_TWIPS / sum(want_tw) for w in want_tw]
    elif sum(least_tw) < COLUMN_TWIPS:
        extra = [w - m for w, m in zip(want_tw, least_tw)]
        room = COLUMN_TWIPS - sum(least_tw)
        widths = [m + room * e / sum(extra) for m, e in zip(least_tw, extra)]
    else:
        widths = [m * COLUMN_TWIPS / sum(least_tw) for m in least_tw]
    widths = [int(round(w)) for w in widths]
    props = sub(table, "tblPr")
    sub(props, "tblW", w=COLUMN_TWIPS, type="dxa")
    sub(props, "tblLayout", type="fixed")
    grid = table.find(q("tblGrid"))
    for column, width in zip(grid.findall(q("gridCol")), widths):
        column.set(q("w"), str(width))
    for row in rows:
        for j, cell in enumerate(row.findall(q("tc"))):
            sub(sub(cell, "tcPr"), "tcW", w=widths[j], type="dxa")
    return widths


def main():
    QA.mkdir(parents=True, exist_ok=True)
    source = package(SOURCE)
    official = package(REFERENCE)
    sr = xml(source["word/document.xml"])
    template = xml(official["word/document.xml"])
    sb = sr.find(q("body"))
    tb = template.find(q("body"))
    reference_paras = tb.findall(q("p"))
    reference_sections = template.findall(".//" + q("sectPr"))
    style_root = xml(official["word/styles.xml"])
    normal = style_root.find("w:style[@w:styleId='Normal']", NS)
    normal_rpr = sub(normal, "rPr")
    sub(normal_rpr, "rFonts", ascii="Times New Roman", hAnsi="Times New Roman",
        eastAsia="Yu Mincho", cs="Times New Roman")
    sub(normal_rpr, "sz", val=20)
    sub(normal_rpr, "szCs", val=20)
    sub(normal_rpr, "lang", val="en-US", eastAsia="ja-JP")
    # Record reference-derived design before filling manuscript slots.
    contract = {
        "reference": str(REFERENCE), "sha256": sha(REFERENCE.read_bytes()),
        "reference_pages": 7, "reference_sections": len(reference_sections),
        "geometry": [etree.tostring(e, encoding="unicode") for e in reference_sections],
        "styles": ["Title", "Normal", "Heading1", "Heading2", "Text"],
        "allowed_changes": [
            "Replace sample body with the current draft's ordered paragraphs and tables.",
            "Use source figures and existing one-column/two-column section transitions.",
            "Use Yu Mincho for Japanese with 0.92 scale rounded to Word half points.",
            "Remove the unassigned manuscript-ID placeholder while retaining page fields.",
            "Set manuscript tables to 8pt Latin and 7.5pt Japanese.",
        ],
        "preserve_only_reference_parts": {
            n: {"bytes": len(d), "sha256": sha(d)} for n, d in official.items()
            if n not in {"word/document.xml", "word/styles.xml",
                         "word/_rels/document.xml.rels", "word/header1.xml",
                         "docProps/core.xml", "[Content_Types].xml"}
        },
        "fidelity_gates": ["All source w:t nodes remain in the same order.",
                           "All eight native tables and nine figures remain.",
                           "Retained official reference remains byte-identical."],
    }
    (QA / "artifact.md").write_text(
        "# IEEE TSG Word template execution contract\n\n" +
        json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
    for child in list(tb):
        tb.remove(child)
    for child in sb:
        tb.append(copy.deepcopy(child))

    rels = xml(official["word/_rels/document.xml.rels"])
    source_rels = {e.get("Id"): e for e in xml(source["word/_rels/document.xml.rels"])}
    used_ids = {v for e in tb.iter() for a, v in e.attrib.items()
                if a.startswith("{" + R + "}")}
    mapped = {}
    output = dict(official)
    for index, old_id in enumerate(sorted(used_ids), 1):
        old = source_rels[old_id]
        if not old.get("Type").endswith("/image"):
            raise ValueError("Review non-image relationship explicitly: " + old_id)
        target = old.get("Target").replace("\\", "/")
        name = f"draft_image_{index:02d}" + Path(target).suffix
        mapped[old_id] = f"rIdDraftImage{index}"
        output["word/media/" + name] = source["word/" + target]
        rel = etree.SubElement(rels, "{" + PKG + "}Relationship")
        rel.set("Id", mapped[old_id])
        rel.set("Type", old.get("Type"))
        rel.set("Target", "media/" + name)
    for element in tb.iter():
        for attribute, value in list(element.attrib.items()):
            if attribute.startswith("{" + R + "}") and value in mapped:
                element.set(attribute, mapped[value])

    roles = Counter()
    reference_ppr = {
        "title": reference_paras[0].find(q("pPr")),
        "author": reference_paras[2].find(q("pPr")),
        "abstract": reference_paras[6].find(q("pPr")),
        "keywords": reference_paras[8].find(q("pPr")),
        "body": reference_paras[13].find(q("pPr")),
    }
    paragraphs = tb.findall(q("p"))
    in_refs = False
    restored = []
    equations = []
    for index, paragraph in enumerate(paragraphs):
        value = text(paragraph)
        old_ppr = paragraph.find(q("pPr"))
        section = copy.deepcopy(old_ppr.find(q("sectPr"))) if (
            old_ppr is not None and old_ppr.find(q("sectPr")) is not None) else None
        images = paragraph.findall(".//{" + A + "}blip")
        if index == 0:
            role, size, style, bold, italic = "title", 24, "Title", False, False
        elif index == 1:
            role, size, style, bold, italic = "author", 11, "Normal", False, False
        elif value.startswith("要旨—"):
            role, size, style, bold, italic = "abstract", 9, "Normal", True, False
        elif value.startswith("索引語—"):
            role, size, style, bold, italic = "keywords", 9, "Normal", True, False
        elif re.match(r"^(?:[IVX]+)?\.\s+", value):
            role, size, style, bold, italic = "section", 10, "Heading1", False, False
        elif re.match(r"^[A-Z]\.\s+", value):
            role, size, style, bold, italic = "subsection", 10, "Heading2", False, True
        elif value == "参考文献":
            in_refs = True
            role, size, style, bold, italic = "references_heading", 10, "Heading1", False, False
        elif in_refs and value:
            role, size, style, bold, italic = "reference", 8, "Normal", False, False
        elif value.startswith("【予定】"):
            role, size, style, bold, italic = "note", 8, "Normal", False, False
        elif value.startswith(("TABLE ", "Fig. ")):
            role, size, style, bold, italic = "caption", 8, "Normal", False, False
        elif images:
            role, size, style, bold, italic = "figure", 10, "Normal", None, None
        elif re.search(r"\([1-5]\)$", value) and re.match(
                r"^(max Σ_t|u_t ∈|b_t −|∀ s ∈ S|r_t =)", value):
            role, size, style, bold, italic = "equation", 10, "Normal", False, True
        elif not value:
            role, size, style, bold, italic = "section_break" if section is not None else "empty", 1, "Normal", None, None
        else:
            role, size, style, bold, italic = "body", 10, "Normal", False, False
        ppr = copy.deepcopy(reference_ppr.get(role, reference_ppr["body"]))
        if ppr is None:
            ppr = etree.Element(q("pPr"))
        for name in ["sectPr", "framePr", "numPr", "tabs", "pStyle", "rPr",
                     "pBdr", "spacing", "ind", "jc", "keepNext", "keepLines",
                     "pageBreakBefore"]:
            for child in ppr.findall(q(name)):
                ppr.remove(child)
        sub(ppr, "pStyle", val=style)
        sub(ppr, "spacing", before=0, after=0, line=240, lineRule="auto")
        sub(ppr, "jc", val="both")
        sub(ppr, "ind", firstLine=144)
        if role in {"title", "author", "section", "references_heading", "equation", "figure"}:
            sub(ppr, "jc", val="center")
            sub(ppr, "ind", firstLine=0)
        if role == "title":
            sub(ppr, "spacing", after=120, line=240, lineRule="auto")
        elif role == "author":
            sub(ppr, "spacing", after=180, line=240, lineRule="auto")
        elif role in {"abstract", "keywords", "note", "caption", "subsection", "reference"}:
            sub(ppr, "ind", firstLine=0)
        if role == "keywords":
            sub(ppr, "spacing", after=120, line=240, lineRule="auto")
        if role in {"section", "subsection", "references_heading"}:
            # Heading styles supply the official before/after spacing.
            ppr.remove(ppr.find(q("spacing")))
            sub(ppr, "keepNext")
        if role == "reference":
            sub(ppr, "ind", left=240, hanging=240)
        if role == "figure":
            sub(ppr, "keepNext")
            sub(ppr, "spacing", before=60, after=0, line=240, lineRule="auto")
        if role == "caption":
            sub(ppr, "spacing", before=0, after=90, line=240, lineRule="auto")
            if value.startswith("TABLE "):
                sub(ppr, "keepNext")
                sub(ppr, "jc", val="center")
            elif value.startswith("Fig. "):
                sub(ppr, "jc", val="both")
        # The separate table-title paragraph immediately precedes each table.
        sibling = paragraph.getnext()
        if sibling is not None and sibling.tag == q("tbl"):
            sub(ppr, "keepNext")
            sub(ppr, "jc", val="center")
            sub(ppr, "ind", firstLine=0)
            size = 8
        if role in {"body", "abstract", "keywords", "note", "caption", "reference", "equation"} or size == 8:
            sub(ppr, "spacing", line=EXACT_LINE[size], lineRule="exact")
        if role == "section" and value.startswith(". "):
            # The saved draft lost the numeral of the first heading ("I. はじめに"
            # in the 17:40 PDF) while its abstract was being edited.
            first = paragraph.find(".//" + q("t"))
            first.text = "I" + first.text
            restored.append(text(paragraph))
        if section is not None:
            ppr.append(section)
        if old_ppr is not None:
            paragraph.remove(old_ppr)
        paragraph.insert(0, ppr)
        split_and_format_runs(paragraph, size, bold, italic)
        roles[role] += 1
        if role == "equation":
            equations.append(paragraph)

    # Keep a run of equations (1)-(4) in one column.
    for paragraph in equations:
        if paragraph.getnext() in equations:
            sub(paragraph.find(q("pPr")), "keepNext")

    for table in tb.findall(q("tbl")):
        for row in table.findall(q("tr")):
            sub(sub(row, "trPr"), "cantSplit")
            for paragraph in row.findall(".//" + q("p")):
                ppr = paragraph.find(q("pPr"))
                if ppr is None:
                    ppr = etree.Element(q("pPr"))
                    paragraph.insert(0, ppr)
                sub(ppr, "spacing", before=0, after=0, line=EXACT_LINE[8], lineRule="exact")
                split_and_format_runs(paragraph, 8)

    table_widths = [fit_table_widths(table) for table in tb.findall(q("tbl"))]

    header_refs = [copy.deepcopy(e) for s in reference_sections
                   for e in s.findall(q("headerReference"))]
    for number, section in enumerate(tb.findall(".//" + q("sectPr"))):
        old_cols = section.find(q("cols"))
        count = int(old_cols.get(q("num"), "1")) if old_cols is not None else 1
        prototype = reference_sections[0 if count == 1 else 1]
        for name in ["pgSz", "pgMar", "cols", "docGrid", "headerReference",
                     "footerReference", "titlePg", "pgNumType"]:
            for old in section.findall(q(name)):
                section.remove(old)
            for new in prototype.findall(q(name)):
                section.append(copy.deepcopy(new))
        if not section.findall(q("headerReference")):
            for header in header_refs:
                section.insert(0, copy.deepcopy(header))
        if number > 0:
            # Only the first section starts the page count at 1; the sections
            # around two-column-wide figures continue it.
            for numbering in section.findall(q("pgNumType")):
                numbering.attrib.pop(q("start"), None)
    if "word/header1.xml" in output:
        header = xml(output["word/header1.xml"])
        for element in header.findall(".//" + q("t")):
            if "REPLACE THIS LINE" in (element.text or ""):
                element.text = ""
        output["word/header1.xml"] = encoded(header)
    output["word/document.xml"] = encoded(template)
    output["word/styles.xml"] = encoded(style_root)
    output["word/_rels/document.xml.rels"] = encoded(rels)
    source_text = sr.xpath(".//w:body//w:t/text()", namespaces=NS)
    final_text = template.xpath(".//w:body//w:t/text()", namespaces=NS)
    expected = "".join(source_text)
    for heading in restored:
        expected = expected.replace(heading[1:], heading, 1)
    assert expected == "".join(final_text), "Body text changed."
    assert len(tb.findall(q("tbl"))) == 8
    assert len(tb.findall(".//{" + A + "}blip")) == 9
    with ZipFile(OUTPUT, "w", compression=ZIP_DEFLATED) as archive:
        for name, data in output.items():
            archive.writestr(name, data)
    report = {
        "reference_sha256": sha(REFERENCE.read_bytes()),
        "source_sha256": sha(SOURCE.read_bytes()), "output": str(OUTPUT),
        "unchanged_body_text_except_restored_headings": True, "tables": 8, "figures": 9,
        "roles": dict(roles), "body_latin_pt": 10, "body_japanese_pt": 9,
        "exact_line_pt": {k: v / 20 for k, v in EXACT_LINE.items()},
        "restored_headings": restored, "table_widths_twips": table_widths,
        "margins_inches": [.7, .7, .65, .65],
        "columns": {"width_inches": 3.5, "gap_inches": .2},
    }
    (QA / "format_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
