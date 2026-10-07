"""Add an editable provenance diagram and numbered poster references.

Run from the repository root: pixi run python poster/build_provenance_flowchart.py
The PNG is a layout proof generated from the same geometry, not a PowerPoint export.
ZIP/XML checks do not establish PowerPoint compatibility; verify new geometry in
PowerPoint before considering a regenerated presentation validated.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
import tempfile
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parent
POSTER = ROOT / "neuron_navigator_poster_refs_updated.pptx"
PREVIEW = ROOT / "provenance_flowchart_preview.png"
GROUP_NAME = "Provenance flowchart"
NS = {
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}
for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)

EMU = 914400
WIDTH, HEIGHT = 9.25, 13.4
ORIGIN = (1.93, 20.63)
INK, BLUE, GREEN, GRAY = "253B4D", "3F7FB7", "4E843D", "728392"

# Text segments are (text, superscript). Positions and dimensions are inches.
NODES = [
    ("software", (0.55, 0.75, 8.1, 0.90),
     [("Neuron Navigator", False), ("6", True), (" + Pixi", False), ("7", True)], 23, "EDF4FA", BLUE),
    ("pfc", (0.4, 2.75, 1.9, 0.80), [("PFC SWCs", False), ("1", True)], 18.5, "EDF4FA", BLUE),
    ("cortex", (2.6, 2.75, 1.9, 0.80), [("Cortex SWCs", False), ("2", True)], 18.5, "EDF4FA", BLUE),
    ("flatmaps", (4.8, 2.75, 1.9, 0.80), [("Flatmaps", False), ("3,4", True)], 18.5, "EDF4FA", BLUE),
    ("fork", (7.0, 2.75, 1.9, 0.80), [("Atlas fork", False), ("5", True)], 18.5, "F2F4F6", GRAY),
    ("neurons", (0.7, 4.6, 3.5, 0.90), [("Neuron Parquet", False)], 22, "F0F6EC", GREEN),
    ("nrrds", (5.0, 4.6, 3.5, 0.90), [("25 µm NRRDs", False), ("5", True)], 22, "F0F6EC", GREEN),
    ("enriched", (2.65, 6.75, 3.9, 0.95), [("Flatmap Parquet", False)], 22, "F0F6EC", GREEN),
    ("atlas", (6.85, 8.10, 2.1, 0.85), [("Allen atlas", False), ("8", True)], 19.5, "EDF4FA", BLUE),
    ("sidecar", (2.65, 9.15, 3.9, 0.95), [("Region sidecar", False)], 23, GREEN, GREEN),
]
# Arrowheads terminate at products; shared junctions keep the two branches clear.
EDGES = [
    ([(0.55, 1.20), (0.15, 1.20), (0.15, 5.05), (0.70, 5.05)], True),
    ([(1.35, 3.55), (1.35, 4.02), (2.45, 4.02)], False),
    ([(3.55, 3.55), (3.55, 4.02), (2.45, 4.02)], False),
    ([(2.45, 4.02), (2.45, 4.60)], True),
    ([(5.75, 3.55), (5.75, 4.02), (6.75, 4.02)], False),
    ([(7.95, 3.55), (7.95, 4.02), (6.75, 4.02)], False),
    ([(6.75, 4.02), (6.75, 4.60)], True),
    ([(2.45, 5.50), (2.45, 6.08), (4.60, 6.08)], False),
    ([(6.75, 5.50), (6.75, 6.08), (4.60, 6.08)], False),
    ([(4.60, 6.08), (4.60, 6.75)], True),
    ([(4.60, 7.70), (4.60, 8.525)], False),
    ([(6.85, 8.525), (4.60, 8.525)], False),
    ([(4.60, 8.525), (4.60, 9.15)], True),
]
LABELS = [
    ("Plugin setup", (0.55, 0.27, 8.1, 0.35), 14, "ctr"),
    ("For region search", (2.65, 10.25, 3.9, 0.35), 16, "ctr"),
]
LEGEND_BOX = (0.15, 10.9, 8.95, 2.45)
LEGEND_TITLE = "Figure 1. Provenance of the flatmap-enabled neuron Parquet and regional-profile sidecar."
LEGEND_TEXT = (
    "CCFv3-registered SWC reconstructions from the prefrontal cortex and whole-cortex "
    "datasets are consolidated into a neuron Parquet using Neuron Navigator. Published "
    "cortical flatmap and depth fields are resampled to a 25 µm atlas grid with the "
    "atlas-enhancement fork and used to append bilateral flatmap coordinates and "
    "cortical depth. Per-neuron summaries derived from Allen atlas annotations are "
    "stored in an associated regional-profile Parquet, linked by file_id, to support "
    "compound anatomical searches. Arrows indicate preparation dependencies; "
    "superscripts identify source references."
)
EXTRA_REFS = [
    "Neuron Navigator. Plugin source code. https://github.com/nimh-dsst/napari-neuron-navigator.",
    "Pixi. Package and environment management. https://pixi.sh/.",
    "Wang Q. et al. (2020). The Allen Mouse Brain Common Coordinate Framework: A 3D Reference Atlas. Cell 181(4), 936–953.e20. DOI: 10.1016/j.cell.2020.04.007.",
    "Hooks B.M. et al. (2018). Topographic precision in sensory and motor corticostriatal projections varies across cell type and cortical area. Nature Communications 9, 3549. DOI: 10.1038/s41467-018-05780-7.",
]
REFERENCE_KEYS = {
    1: "10.12412/BSDC.1690164952.20001",
    2: "10.12412/BSDC.1747279998.20001",
    3: "10.1162/imag_a_00209",
    4: "10.5281/zenodo.11218079",
    5: "github.com/joshlawrimore/atlas-enhancement",
    6: "github.com/nimh-dsst/napari-neuron-navigator",
    7: "pixi.sh/",
    8: "10.1016/j.cell.2020.04.007",
    9: "10.1038/s41467-018-05780-7",
}


def verify_references(body, group):
    """Check numbered sources against both documents and every bubble citation."""
    workflow = (ROOT / "neuron_navigator_MOs_poster_workflow.md").read_text()
    assert LEGEND_TITLE in workflow and LEGEND_TEXT in workflow, "Legend must match the workflow"
    bibliography = workflow.split("# REFERENCES / SOURCE PROVENANCE", 1)[1]
    workflow_refs = dict(re.findall(r"(?ms)^(\d+)\. (.*?)(?=^\d+\. |^---|\Z)", bibliography))
    paragraphs = body.findall("a:p", NS)
    assert len(paragraphs) == len(REFERENCE_KEYS)
    for number, key in REFERENCE_KEYS.items():
        text = "".join(t.text or "" for t in paragraphs[number - 1].findall(".//a:t", NS))
        assert text.startswith(f"{number}. ") and key in text, (number, "Poster reference mismatch")
        assert key in workflow_refs[str(number)], (number, "Workflow reference mismatch")
    for run in group.findall(".//a:r", NS):
        if run.find("a:rPr", NS).get("baseline") == "30000":
            assert all(int(n) in REFERENCE_KEYS for n in run.find("a:t", NS).text.split(","))
    text = " ".join(t.text or "" for t in group.findall(".//a:t", NS))
    assert re.search(r"\b0[A-H]\b", text) is None, "Step numbers must not appear in the diagram"


def element(parent, tag, **attrs):
    prefix, local = tag.split(":")
    return ET.SubElement(parent, f"{{{NS[prefix]}}}{local}", {k: str(v) for k, v in attrs.items()})


def emu(value):
    return round(value * EMU)


def transform(parent, box, *, group=False):
    x, y, w, h = box
    xf = element(parent, "a:xfrm")
    element(xf, "a:off", x=emu(x), y=emu(y))
    element(xf, "a:ext", cx=max(1, emu(w)), cy=max(1, emu(h)))
    if group:
        element(xf, "a:chOff", x=0, y=0)
        element(xf, "a:chExt", cx=emu(w), cy=emu(h))


def fill(parent, color):
    element(element(parent, "a:solidFill"), "a:srgbClr", val=color)


def paragraph(body, segments, size, *, color=INK, align="ctr", bold=False):
    p = element(body, "a:p")
    pp = element(p, "a:pPr", algn=align)
    element(pp, "a:buNone")
    for text, superscript in segments:
        run = element(p, "a:r")
        props = element(run, "a:rPr", lang="en-US", sz=round(size * 100 * (0.65 if superscript else 1)), b=int(bold))
        if superscript:
            props.set("baseline", "30000")
        fill(props, color)
        element(props, "a:latin", typeface="Arial")
        element(run, "a:t").text = text
    element(p, "a:endParaRPr", lang="en-US", sz=round(size * 100))


def add_shape(group, shape_id, name, box, segments, size, *, background=None, border=GRAY, align="ctr"):
    shape = element(group, "p:sp")
    nv = element(shape, "p:nvSpPr")
    element(nv, "p:cNvPr", id=shape_id, name=name)
    element(nv, "p:cNvSpPr")
    element(nv, "p:nvPr")
    props = element(shape, "p:spPr")
    transform(props, box)
    geom = element(props, "a:prstGeom", prst="roundRect" if background else "rect")
    element(geom, "a:avLst")
    if background:
        fill(props, background)
        line = element(props, "a:ln", w=19050)
        fill(line, border)
    else:
        element(props, "a:noFill")
        element(element(props, "a:ln"), "a:noFill")
    body = element(shape, "p:txBody")
    element(body, "a:bodyPr", wrap="none", lIns=emu(0.07), rIns=emu(0.07), tIns=0, bIns=0, anchor="ctr")
    element(body, "a:lstStyle")
    paragraph(body, segments, size, color="FFFFFF" if name == "sidecar" else INK if background else GRAY, align=align, bold=background is not None)
    return shape


def add_legend(group, shape_id):
    shape = add_shape(group, shape_id, "Provenance figure legend", LEGEND_BOX,
                      [(LEGEND_TITLE, False)], 15, align="l")
    body = shape.find("p:txBody", NS)
    props = body.find("a:bodyPr", NS)
    props.set("wrap", "square")
    props.set("anchor", "t")
    title = body.find("a:p/a:r/a:rPr", NS)
    title.set("b", "1")
    title.find("a:solidFill/a:srgbClr", NS).set("val", INK)
    pp = body.find("a:p/a:pPr", NS)
    after = element(pp, "a:spcAft")
    element(after, "a:spcPts", val="500")
    pp.remove(after)
    pp.insert(0, after)
    paragraph(body, [(LEGEND_TEXT, False)], 15, color=INK, align="l")


def add_edge(group, shape_id, points, arrow):
    """Use ordinary straight connectors, one per orthogonal line segment."""
    assert len(points) == 2
    connector = element(group, "p:cxnSp")
    nv = element(connector, "p:nvCxnSpPr")
    element(nv, "p:cNvPr", id=shape_id, name=f"Provenance arrow {shape_id}")
    element(nv, "p:cNvCxnSpPr")
    element(nv, "p:nvPr")
    props = element(connector, "p:spPr")
    minx, miny = min(x for x, _ in points), min(y for _, y in points)
    w, h = max(x for x, _ in points) - minx, max(y for _, y in points) - miny
    xf = element(props, "a:xfrm")
    if points[-1][0] < points[0][0]:
        xf.set("flipH", "1")
    if points[-1][1] < points[0][1]:
        xf.set("flipV", "1")
    element(xf, "a:off", x=emu(minx), y=emu(miny))
    element(xf, "a:ext", cx=emu(w), cy=emu(h))
    element(element(props, "a:prstGeom", prst="line"), "a:avLst")
    line = element(props, "a:ln", w=19050)
    fill(line, GRAY)
    element(line, "a:round")
    if arrow:
        element(line, "a:tailEnd", type="triangle", w="med", len="med")


def preview():
    scale = 130
    image = Image.new("RGB", (round(WIDTH * scale), round(HEIGHT * scale)), "white")
    draw = ImageDraw.Draw(image)
    regular = font_manager.findfont(font_manager.FontProperties(family="Arial"))
    bold = font_manager.findfont(font_manager.FontProperties(family="Arial", weight="bold"))
    for points, arrow in EDGES:
        pixels = [(round(x * scale), round(y * scale)) for x, y in points]
        draw.line(pixels, fill=f"#{GRAY}", width=3, joint="curve")
        if arrow:
            (ax, ay), (bx, by) = pixels[-2:]
            dx, dy = bx - ax, by - ay
            length = (dx * dx + dy * dy) ** 0.5
            ux, uy = dx / length, dy / length
            draw.polygon([(bx, by), (bx - 12 * ux + 5 * uy, by - 12 * uy - 5 * ux), (bx - 12 * ux - 5 * uy, by - 12 * uy + 5 * ux)], fill=f"#{GRAY}")
    for name, (x, y, w, h), segments, size, background, border in NODES:
        draw.rounded_rectangle((x * scale, y * scale, (x + w) * scale, (y + h) * scale), radius=0.14 * scale, fill=f"#{background}", outline=f"#{border}", width=3)
        runs = [(text, sup, ImageFont.truetype(bold, round(size * scale / 72 * (0.65 if sup else 1)))) for text, sup in segments]
        widths = [draw.textlength(text, font=font) for text, _, font in runs]
        assert sum(widths) < (w - 0.14) * scale, (name, "Text too wide")
        cursor = (x + w / 2) * scale - sum(widths) / 2
        baseline = (y + h / 2) * scale + 0.35 * size * scale / 72
        for (text, sup, font), width in zip(runs, widths):
            draw.text((cursor, baseline - (0.4 * size * scale / 72 if sup else 0)), text, font=font, fill="white" if name == "sidecar" else f"#{INK}", anchor="ls")
            cursor += width
    for text, (x, y, w, h), size, align in LABELS:
        font = ImageFont.truetype(regular, round(size * scale / 72))
        draw.text(((x + (w / 2 if align == "ctr" else 0)) * scale, (y + h / 2) * scale), text, font=font, fill=f"#{GRAY}", anchor="mm" if align == "ctr" else "lm")
    x, y, w, h = LEGEND_BOX
    cursor_y = y * scale
    for content, face in [(LEGEND_TITLE, bold), (LEGEND_TEXT, regular)]:
        font = ImageFont.truetype(face, round(15 * scale / 72))
        lines, line = [], ""
        for word in content.split():
            candidate = (line + " " + word).strip()
            if draw.textlength(candidate, font=font) > (w - 0.14) * scale:
                lines.append(line)
                line = word
            else:
                line = candidate
        lines.append(line)
        for line in lines:
            draw.text(((x + 0.07) * scale, cursor_y), line, font=font, fill=f"#{INK}", anchor="lt")
            cursor_y += 18 * scale / 72
        cursor_y += 5 * scale / 72
    assert cursor_y - 5 * scale / 72 <= (y + h) * scale, "Legend exceeds its textbox"
    image.save(PREVIEW)


def main():
    # Render and check text fit before changing the presentation.
    preview()
    with ZipFile(POSTER) as archive:
        infos = archive.infolist()
        data = {info.filename: archive.read(info.filename) for info in infos}
    root = ET.fromstring(data["ppt/slides/slide1.xml"])
    tree = root.find("p:cSld/p:spTree", NS)
    for child in list(tree):
        prop = child.find("p:nvGrpSpPr/p:cNvPr", NS)
        if prop is not None and prop.get("name") in (GROUP_NAME, "Step 0 provenance flowchart"):
            tree.remove(child)
    next_id = max(int(p.get("id")) for p in root.findall(".//p:cNvPr", NS)) + 1
    group = element(tree, "p:grpSp")
    nv = element(group, "p:nvGrpSpPr")
    element(nv, "p:cNvPr", id=next_id, name=GROUP_NAME, descr="Step 0A–0H source and preparation dependencies; a provenance schematic, not an analysis result.")
    element(nv, "p:cNvGrpSpPr")
    element(nv, "p:nvPr")
    transform(element(group, "p:grpSpPr"), (*ORIGIN, WIDTH, HEIGHT), group=True)
    next_id += 1
    for points, arrow in EDGES:
        for index in range(len(points) - 1):
            add_edge(group, next_id, points[index:index + 2], arrow and index == len(points) - 2)
            next_id += 1
    for name, box, segments, size, background, border in NODES:
        add_shape(group, next_id, name, box, segments, size, background=background, border=border)
        next_id += 1
    for label, box, size, align in LABELS:
        add_shape(group, next_id, "Provenance caption " + label, box, [(label, False)], size, align=align)
        next_id += 1
    add_legend(group, next_id)
    references = next(s for s in tree.findall("p:sp", NS) if s.find("p:nvSpPr/p:cNvPr", NS).get("name") == "TextBox 21")
    body = references.find("p:txBody", NS)
    paragraphs = body.findall("a:p", NS)
    assert len(paragraphs) in (5, 8, 9), "Unexpected poster references; review numbering before regeneration."
    if len(paragraphs) < len(REFERENCE_KEYS):
        for text in EXTRA_REFS[len(paragraphs) - 5:]:
            p = deepcopy(paragraphs[-1])
            runs = p.findall("a:r", NS)
            for run in runs[1:]:
                p.remove(run)
            runs[0].find("a:t", NS).text = text
            body.append(p)
    for index, p in enumerate(body.findall("a:p", NS), 1):
        first = p.find("a:r/a:t", NS)
        first.text = f"{index}. " + re.sub(r"^\d+\. ", "", first.text)
        p.find("a:pPr/a:spcAft/a:spcPts", NS).set("val", "600" if index < len(REFERENCE_KEYS) else "0")
    verify_references(body, group)
    ids = [p.get("id") for p in root.findall(".//p:cNvPr", NS)]
    assert len(ids) == len(set(ids))
    data["ppt/slides/slide1.xml"] = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    notes = ET.fromstring(data["ppt/notesSlides/notesSlide1.xml"])
    notes_body = next(s.find("p:txBody", NS) for s in notes.findall(".//p:sp", NS) if s.find("p:nvSpPr/p:nvPr/p:ph", NS) is not None and s.find("p:nvSpPr/p:nvPr/p:ph", NS).get("type") == "body")
    for p in list(notes_body.findall("a:p", NS)):
        if "Step 0 provenance:" in "".join(p.itertext()):
            notes_body.remove(p)
    paragraph(notes_body, [("Step 0 provenance: Diagram added and rebuilt 2026-10-07 from workflow Steps 0A–0H; step numbers are omitted from the diagram. References 1–2: SWC datasets; 3–4: flatmap paper and Zenodo v4.1; 5: atlas-enhancement fork and the 25 µm resampling utility at ea735eacdfe8afc40d006d80caefd54d6b30fdd3; 6: Neuron Navigator; 7: Pixi; 8: Allen CCFv3 (atlas obtained through BrainGlobe). Neuron Parquet is the plugin SWC-conversion output. 25 µm NRRDs feed whole-Parquet flatmap/depth augmentation; the Allen atlas also supplies their target grid. The regional sidecar is built from the chosen neuron Parquet and compatible atlas; flatmap augmentation is part of this poster's preparation order, not a requirement for all regional sidecars. Pixi also manages the fork's separate preparation environment. This schematic records dependencies, not completed downloads or new analysis results. See workflow for exact parameters, strict versus harmonic provenance, hashes, and remaining run-record fields.", False)], 12, align="l")
    data["ppt/notesSlides/notesSlide1.xml"] = ET.tostring(notes, encoding="utf-8", xml_declaration=True)
    with tempfile.NamedTemporaryFile(prefix="neuron_poster_before_flowchart_", suffix=".pptx", delete=False) as backup:
        backup.write(POSTER.read_bytes())
        backup_path = backup.name
    temporary = POSTER.with_suffix(".updating.pptx")
    with ZipFile(temporary, "w") as output:
        for info in infos:
            output.writestr(info, data[info.filename])
    with ZipFile(temporary) as check:
        assert check.testzip() is None
        for name in check.namelist():
            if name.endswith((".xml", ".rels")):
                ET.fromstring(check.read(name))
    temporary.replace(POSTER)
    print(f"Updated {POSTER.name}; editable diagram, {len(REFERENCE_KEYS)} numbered references.")
    print(f"Layout proof: {PREVIEW}")
    print(f"Original presentation backup: {backup_path}")


if __name__ == "__main__":
    main()
