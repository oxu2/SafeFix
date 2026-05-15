#!/usr/bin/env python3
"""Build an illustrative Draw.io framework figure for AttriSelect."""

from __future__ import annotations

import base64
import io
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "frame" / "attriselect_framework.drawio"
THUMB_DIR = ROOT / "frame" / "assets" / "attriselect_framework_thumbnails"

PAGE_W = 1785
PAGE_H = 815
FONT = "Arial"
Y_OFFSET = -45

COL = {
    "ink": "#111827",
    "muted": "#475569",
    "line": "#CBD5E1",
    "blue": "#3A6EA5",
    "blue_fill": "#E8F1FA",
    "green": "#2A9D8F",
    "green_fill": "#E8F5E9",
    "red": "#DC2626",
    "red_fill": "#FDECEC",
    "orange": "#F97316",
    "orange_fill": "#FFF7ED",
    "amber": "#D4A252",
    "amber_fill": "#FFF4D6",
    "purple": "#7C3AED",
    "purple_fill": "#F3E8FF",
    "gray_fill": "#F8FAFC",
}

SOURCE_POOL = [
    "figures/teaser_imagenet_selection/ours_unique_diverse/01_id24_backpack_purple_fabric_snow.jpeg",
    "figures/teaser_imagenet_selection/ours_unique_diverse/09_id2677_coffee_mug_yellow_ceramic_house.jpeg",
    "figures/teaser_imagenet_selection/ours_unique_diverse/17_id6289_electric_guitar_silver_metallic_sky.jpeg",
    "figures/teaser_imagenet_selection/traditional_al_common/13_id3901_desk_white_wooden_wall.jpeg",
]

OURS = [
    "figures/teaser_imagenet_selection/ours_unique_fabric/01_id3_backpack_black_fabric_wild.jpeg",
    "figures/teaser_imagenet_selection/ours_unique_fabric/02_id3534_coffee_mug_purple_fabric_house.jpeg",
    "figures/teaser_imagenet_selection/ours_unique_fabric/04_id11012_rocking_chair_maroon_fabric_trees.jpeg",
    "figures/teaser_imagenet_selection/ours_unique_fabric/05_id24_backpack_purple_fabric_snow.jpeg",
    "figures/teaser_imagenet_selection/ours_unique_fabric/16_id9543_purse_pink_fabric_bridge.jpeg",
    "figures/teaser_imagenet_selection/ours_unique_fabric/18_id9717_purse_orange_fabric_street.jpeg",
]

TRADITIONAL_AL = [
    "figures/teaser_imagenet_selection/traditional_al_common/01_id171_backpack_black_plastic_wall.jpeg",
    "figures/teaser_imagenet_selection/traditional_al_common/09_id2600_coffee_mug_white_ceramic_wall.jpeg",
    "figures/teaser_imagenet_selection/traditional_al_common/13_id3901_desk_white_wooden_wall.jpeg",
    "figures/teaser_imagenet_selection/traditional_al_common/17_id5202_electric_guitar_black_plastic_wall.jpeg",
    "figures/teaser_imagenet_selection/traditional_al_common/25_id8198_pitcher_white_ceramic_wall.jpeg",
    "figures/teaser_imagenet_selection/traditional_al_common/29_id9133_purse_brown_leather_wall.jpeg",
]

GEN_CLASSES = ["backpack", "coffee_mug", "purse", "rocking_chair", "water_bottle", "electric_guitar"]


def generated_paths() -> list[str]:
    base = ROOT / "data" / "pinkcolor_fabrictexture_instruct_pix2pix_seed42_1k"
    paths: list[str] = []
    for cls in GEN_CLASSES:
        candidates = sorted((base / cls).glob("*.jpg"))
        if candidates:
            paths.append(str(candidates[0].relative_to(ROOT)))
    return paths


def style(parts: list[str] | tuple[str, ...]) -> str:
    return ";".join(parts) + ";"


def center_crop(image: Image.Image, target: tuple[int, int]) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    tw, th = target
    ratio = tw / th
    w, h = image.size
    if w / h > ratio:
        nw = int(h * ratio)
        left = (w - nw) // 2
        image = image.crop((left, 0, left + nw, h))
    else:
        nh = int(w / ratio)
        top = (h - nh) // 2
        image = image.crop((0, top, w, top + nh))
    return image.resize(target, Image.Resampling.LANCZOS)


def image_uri(path: str, key: str, size: tuple[int, int]) -> str:
    src = ROOT / path
    if not src.exists():
        raise FileNotFoundError(src)
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    thumb = center_crop(Image.open(src), size)
    thumb.save(THUMB_DIR / f"{key}.jpg", format="JPEG", quality=84, optimize=True)
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=84, optimize=True)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    # Draw.io uses semicolon as a style separator; use the comma-only data URI
    # form so the embedded image remains a single style value.
    return f"data:image/jpeg,{encoded}"


class Diagram:
    def __init__(self) -> None:
        self.mxfile = ET.Element("mxfile", {"host": "drawio", "version": "26.0.0"})
        diagram = ET.SubElement(
            self.mxfile,
            "diagram",
            {"name": "AttriSelect Framework", "id": "attriselect-framework"},
        )
        self.graph = ET.SubElement(
            diagram,
            "mxGraphModel",
            {
                "dx": str(PAGE_W),
                "dy": str(PAGE_H),
                "grid": "1",
                "gridSize": "10",
                "guides": "1",
                "tooltips": "1",
                "connect": "1",
                "arrows": "1",
                "fold": "1",
                "page": "1",
                "pageScale": "1",
                "pageWidth": str(PAGE_W),
                "pageHeight": str(PAGE_H),
                "math": "0",
                "shadow": "0",
            },
        )
        self.root = ET.SubElement(self.graph, "root")
        ET.SubElement(self.root, "mxCell", {"id": "0"})
        ET.SubElement(self.root, "mxCell", {"id": "1", "parent": "0"})

    def cell(
        self,
        cell_id: str,
        value: str,
        st: str,
        x: float,
        y: float,
        w: float,
        h: float,
        parent: str = "1",
    ) -> None:
        cell = ET.SubElement(
            self.root,
            "mxCell",
            {"id": cell_id, "value": value, "style": st, "vertex": "1", "parent": parent},
        )
        ET.SubElement(
            cell,
            "mxGeometry",
            {
                "x": str(x),
                "y": str(y if cell_id == "page_bg" else y + Y_OFFSET),
                "width": str(w),
                "height": str(h),
                "as": "geometry",
            },
        )

    def edge(self, edge_id: str, source: str, target: str, color: str = "#111827", dashed: bool = False) -> None:
        st = style(
            [
                "edgeStyle=orthogonalEdgeStyle",
                "rounded=0",
                "orthogonalLoop=1",
                "jettySize=auto",
                "html=1",
                "endArrow=block",
                "endFill=1",
                f"strokeColor={color}",
                "strokeWidth=3",
                f"fontFamily={FONT}",
                "fontSize=16",
                "dashed=1" if dashed else "dashed=0",
            ]
        )
        cell = ET.SubElement(
            self.root,
            "mxCell",
            {"id": edge_id, "value": "", "style": st, "edge": "1", "parent": "1", "source": source, "target": target},
        )
        ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})

    def write(self) -> None:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        tree = ET.ElementTree(self.mxfile)
        tree.write(OUT, encoding="UTF-8", xml_declaration=True)


def text_style(size: int, bold: bool = False, color: str = COL["ink"], align: str = "center") -> str:
    return style(
        [
            "text",
            "html=1",
            "strokeColor=none",
            "fillColor=none",
            f"align={align}",
            "verticalAlign=middle",
            "whiteSpace=wrap",
            f"fontSize={size + 2}",
            f"fontStyle={1 if bold else 0}",
            f"fontFamily={FONT}",
            f"fontColor={color}",
        ]
    )


def box_style(fill: str, stroke: str, width: int = 2, arc: int = 10) -> str:
    return style(
        [
            "rounded=1",
            "whiteSpace=wrap",
            "html=1",
            f"arcSize={arc}",
            f"fillColor={fill}",
            f"strokeColor={stroke}",
            f"strokeWidth={width}",
            "fontSize=19",
            f"fontFamily={FONT}",
            f"fontColor={COL['ink']}",
            "align=center",
            "verticalAlign=middle",
        ]
    )


def image_style(uri: str, stroke: str) -> str:
    return style(
        [
            "shape=image",
            "html=1",
            "imageAspect=1",
            "aspect=fixed",
            "whiteSpace=wrap",
            "rounded=1",
            "arcSize=4",
            f"strokeColor={stroke}",
            "strokeWidth=2",
            f"image={uri}",
        ]
    )


def add_panel(d: Diagram, cid: str, title: str, x: int, y: int, w: int, h: int, fill: str, stroke: str) -> None:
    d.cell(cid, "", box_style(fill, stroke, 3, 12), x, y, w, h)
    d.cell(f"{cid}_title", title, text_style(22, True), x + 20, y + 14, w - 40, 34)


def add_badge(d: Diagram, cid: str, label: str, x: int, y: int, w: int, color: str, fill: str) -> None:
    d.cell(
        cid,
        label,
        style(
            [
                "rounded=1",
                "whiteSpace=wrap",
                "html=1",
                "arcSize=40",
                f"fillColor={fill}",
                f"strokeColor={color}",
                "strokeWidth=2",
                "fontSize=18",
                "fontStyle=1",
                f"fontFamily={FONT}",
                f"fontColor={color}",
                "align=center",
                "verticalAlign=middle",
            ]
        ),
        x,
        y,
        w,
        36,
    )


def add_arrow(d: Diagram, cid: str, x: int, y: int, w: int, h: int, color: str, direction: str = "east") -> None:
    d.cell(
        cid,
        "",
        style(
            [
                "shape=singleArrow",
                "whiteSpace=wrap",
                "html=1",
                f"direction={direction}",
                f"fillColor={color}",
                f"strokeColor={color}",
                "strokeWidth=1",
            ]
        ),
        x,
        y,
        w,
        h,
    )


def add_thumbnail_row(
    d: Diagram,
    prefix: str,
    paths: list[str],
    x: int,
    y: int,
    w: int,
    h: int,
    gap: int,
    stroke: str,
) -> list[str]:
    ids: list[str] = []
    for i, path in enumerate(paths):
        uri = image_uri(path, f"{prefix}_{i + 1:02d}", (w, h))
        cid = f"{prefix}_{i + 1:02d}"
        d.cell(cid, "", image_style(uri, stroke), x + i * (w + gap), y, w, h)
        ids.append(cid)
    return ids


def add_thumbnail_grid(
    d: Diagram,
    prefix: str,
    paths: list[str],
    x: int,
    y: int,
    w: int,
    h: int,
    cols: int,
    gap: int,
    stroke: str,
) -> list[str]:
    ids: list[str] = []
    for i, path in enumerate(paths):
        row, col = divmod(i, cols)
        uri = image_uri(path, f"{prefix}_{i + 1:02d}", (w, h))
        cid = f"{prefix}_{i + 1:02d}"
        d.cell(cid, "", image_style(uri, stroke), x + col * (w + gap), y + row * (h + gap), w, h)
        ids.append(cid)
    return ids


def add_bar_chart(d: Diagram, x: int, y: int, before: bool) -> None:
    d.cell(
        f"chart_box_{x}_{y}",
        "",
        style(
            [
                "rounded=1",
                "whiteSpace=wrap",
                "html=1",
                "arcSize=3",
                "fillColor=#FFFFFF",
                f"strokeColor={COL['line']}",
                "strokeWidth=2",
            ]
        ),
        x,
        y,
        210,
        118,
    )
    d.cell(f"chart_title_{x}_{y}", "Attribute-slice performance", text_style(14, True), x + 8, y + 8, 194, 20)
    values = [74, 58, 24 if before else 68, 52, 30 if before else 71, 62, 55]
    colors = [COL["blue"], COL["blue"], COL["red"] if before else "#22C55E", COL["blue"], COL["red"] if before else "#22C55E", COL["blue"], COL["blue"]]
    bx = x + 24
    base = y + 93
    for i, (val, color) in enumerate(zip(values, colors)):
        bw = 14
        bh = val * 0.75
        d.cell(
            f"bar_{x}_{y}_{i}",
            "",
            style(
                [
                    "rounded=0",
                    "whiteSpace=wrap",
                    "html=1",
                    f"fillColor={color}",
                    "strokeColor=#0F172A",
                    "strokeWidth=1",
                ]
            ),
            bx + i * 22,
            base - bh,
            bw,
            bh,
        )
    d.cell(f"chart_low_{x}_{y}", "fabric", text_style(12, True, COL["red"] if before else "#15803D"), x + 64, y + 94, 46, 18)
    d.cell(f"chart_low2_{x}_{y}", "pink", text_style(12, True, COL["red"] if before else "#15803D"), x + 108, y + 94, 42, 18)


def add_model_cube(d: Diagram, x: int, y: int) -> None:
    d.cell("model_front", "Vision<br>model", box_style("#F5C19C", "#B45309", 3, 4), x, y + 28, 150, 92)
    d.cell(
        "model_top",
        "",
        style(["shape=parallelogram", "perimeter=parallelogramPerimeter", "whiteSpace=wrap", "html=1", "fillColor=#FED7AA", "strokeColor=#B45309", "strokeWidth=2", "direction=east"]),
        x + 24,
        y,
        150,
        36,
    )
    d.cell(
        "model_side",
        "",
        style(["shape=parallelogram", "perimeter=parallelogramPerimeter", "whiteSpace=wrap", "html=1", "fillColor=#EA9867", "strokeColor=#B45309", "strokeWidth=2", "direction=north"]),
        x + 150,
        y + 28,
        38,
        92,
    )


def build() -> None:
    gen = generated_paths()
    gen_show = gen[:2]
    d = Diagram()

    d.cell("page_bg", "", box_style("#FFFFFF", "#FFFFFF", 0), 0, 0, PAGE_W, PAGE_H)

    add_panel(d, "diag_panel", "1  Diagnose an attribute failure", 35, 70, 430, 375, "#F8FAFC", "#334155")
    add_panel(d, "gen_panel", "2  Generate targeted repair data", 35, 475, 430, 355, "#FFFBEB", "#D97706")
    add_panel(d, "select_panel", "3  Select real source companions", 495, 70, 895, 760, "#EFF6FF", "#3A6EA5")
    add_panel(d, "repair_panel", "4  Retrain and repair", 1420, 70, 345, 760, "#F0FDF4", "#16A34A")

    # Diagnosis panel: image examples, attribute tags, and a small failure chart.
    d.cell("source_label", "source pool", text_style(17, True), 60, 112, 150, 24)
    add_thumbnail_grid(d, "source_pool", SOURCE_POOL, 60, 142, 72, 72, 2, 10, COL["blue"])
    d.cell("lvlm_box", "<b>LVLM</b><br>attribute<br>assignment", box_style("#FFFFFF", COL["green"], 2), 235, 135, 180, 78)
    add_badge(d, "tag_color", "pink", 245, 235, 88, COL["purple"], COL["purple_fill"])
    add_badge(d, "tag_texture", "fabric", 348, 235, 92, COL["red"], COL["red_fill"])
    add_bar_chart(d, 64, 320, before=True)
    d.cell(
        "bug_slice",
        "<b>Failure slice</b><br><font color=\"#DC2626\">pink + fabric</font>",
        box_style(COL["red_fill"], COL["red"], 3),
        292,
        337,
        140,
        70,
    )
    add_arrow(d, "arrow_source_lvlm", 206, 178, 34, 18, COL["ink"], "east")
    add_arrow(d, "arrow_lvlm_chart", 226, 278, 22, 52, COL["muted"], "south")
    add_arrow(d, "arrow_chart_bug", 260, 363, 34, 18, COL["red"], "east")

    # Generation panel.
    d.cell("bug_seed", "<b>diagnosed slice</b><br>pink + fabric", box_style(COL["red_fill"], COL["red"], 2), 65, 565, 135, 70)
    d.cell("generator", "<b>Conditional<br>generator</b>", box_style("#DBEAFE", COL["blue"], 3), 245, 555, 140, 86)
    add_thumbnail_row(d, "gen_imgs", gen_show, 210, 690, 76, 76, 14, COL["orange"])
    d.cell("gen_pool", "<b>generated repair images</b><br>pink + fabric", text_style(15, True, COL["ink"]), 145, 776, 280, 54)
    add_arrow(d, "arrow_bug_gen", 202, 592, 42, 20, COL["red"], "east")
    add_arrow(d, "arrow_gen_imgs", 300, 642, 22, 40, COL["orange"], "south")
    add_arrow(d, "arrow_gen_to_select", 458, 748, 42, 18, COL["orange"], "east")

    # Selection panel: visual source/gen inputs, three compact score cues, and
    # side-by-side selected-set evidence.
    d.cell("input_source", "annotated source candidates", text_style(17, True), 525, 122, 230, 24)
    add_thumbnail_grid(d, "cand", OURS[:3] + TRADITIONAL_AL[:3], 535, 154, 66, 54, 3, 9, COL["blue"])
    d.cell("input_gen", "generated repair data", text_style(17, True), 775, 122, 210, 24)
    add_thumbnail_row(d, "sel_gen", gen_show, 820, 158, 66, 66, 14, COL["orange"])
    d.cell("attriselect_core", "<b>AttriSelect</b><br>attribute-aware source selector", box_style("#FFFFFF", COL["green"], 3), 1045, 142, 270, 80)

    component_y = 270
    component_specs = [
        ("rare_card", "1", "rare within class", COL["green"], COL["green_fill"], 535),
        ("comp_card", "2", "avoid generated redundancy", COL["red"], COL["red_fill"], 820),
        ("bal_card", "3", "keep class coverage", COL["amber"], COL["amber_fill"], 1105),
    ]
    for cid, num, label, stroke, fill, x in component_specs:
        d.cell(cid, f"<b>{num}. {label}</b>", box_style(fill, stroke, 3), x, component_y, 235, 78)

    add_arrow(d, "arrow_cand_core", 755, 178, 42, 18, COL["blue"], "east")
    add_arrow(d, "arrow_gen_core", 1002, 178, 42, 18, COL["orange"], "east")
    add_arrow(d, "arrow_core_cards", 1164, 222, 20, 44, COL["muted"], "south")

    d.cell("ours_group", "", style(["rounded=1", "whiteSpace=wrap", "html=1", "arcSize=8", f"fillColor={COL['green_fill']}", f"strokeColor={COL['green']}", "strokeWidth=2", "dashed=1"]), 520, 382, 830, 168)
    d.cell("ours_strip_label", "<b>AttriSelect selects</b>: fabric-related real images, diverse classes/backgrounds", text_style(16, True, COL["green"], "left"), 545, 394, 720, 24)
    add_thumbnail_row(d, "ours_keep", OURS, 545, 430, 80, 62, 12, COL["green"])
    add_badge(d, "ours_badge_1", "fabric cues", 545, 505, 105, COL["green"], "#FFFFFF")
    add_badge(d, "ours_badge_2", "class spread", 660, 505, 122, COL["green"], "#FFFFFF")
    add_badge(d, "ours_badge_3", "scene spread", 792, 505, 124, COL["green"], "#FFFFFF")
    d.cell("selected_box", "<b>selected source subset</b>", box_style("#FFFFFF", COL["green"], 3), 1075, 444, 260, 62)
    add_arrow(d, "arrow_cards_ours", 1235, 348, 18, 36, COL["green"], "south")

    d.cell("al_group", "", style(["rounded=1", "whiteSpace=wrap", "html=1", "arcSize=8", f"fillColor={COL['orange_fill']}", f"strokeColor={COL['orange']}", "strokeWidth=2", "dashed=1"]), 520, 588, 830, 170)
    d.cell("al_strip_label", "<b>Generic AL often selects</b>: common backgrounds/materials", text_style(16, True, COL["orange"], "left"), 545, 600, 720, 24)
    add_thumbnail_row(d, "al_keep", TRADITIONAL_AL, 545, 636, 80, 62, 12, COL["orange"])
    add_badge(d, "al_badge_1", "wall", 545, 711, 70, COL["orange"], "#FFFFFF")
    add_badge(d, "al_badge_2", "common materials", 625, 711, 168, COL["orange"], "#FFFFFF")
    add_badge(d, "al_badge_3", "head values", 803, 711, 116, COL["orange"], "#FFFFFF")

    # Repair panel.
    d.cell("repair_set", "<b>repair training set</b>", box_style("#FFFFFF", "#64748B", 2), 1475, 122, 235, 58)
    add_thumbnail_row(d, "repair_src", OURS[:2], 1470, 220, 62, 58, 12, COL["green"])
    d.cell("plus", "+", text_style(28, True, COL["muted"]), 1610, 225, 30, 30)
    add_thumbnail_grid(d, "repair_gen", gen_show, 1645, 215, 54, 50, 1, 8, COL["orange"])
    add_model_cube(d, 1495, 372)
    add_bar_chart(d, 1477, 565, before=False)
    d.cell("smile", "OK", text_style(20, True, "#16A34A"), 1668, 565, 44, 36)
    d.cell("repaired", "<b>repaired model</b><br>higher target-slice performance", box_style(COL["green_fill"], COL["green"], 3), 1478, 724, 235, 70)
    add_arrow(d, "arrow_repair_to_model", 1585, 318, 18, 48, COL["ink"], "south")
    add_arrow(d, "arrow_model_to_chart", 1585, 496, 18, 56, COL["green"], "south")
    add_arrow(d, "arrow_chart_to_done", 1585, 688, 18, 34, COL["green"], "south")
    add_arrow(d, "arrow_select_to_repair", 1384, 474, 42, 18, COL["green"], "east")

    d.write()


if __name__ == "__main__":
    build()
