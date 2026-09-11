#!/usr/bin/env python3
"""Comment duplicate-title BibTeX aliases and retain one canonical entry."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path


def normalized_title(block: str) -> str:
    match = re.search(
        r"(?ims)^\s*title\s*=\s*(?:\{(.*?)\}|\"(.*?)\")\s*,?\s*$",
        block,
    )
    if not match:
        return ""
    title = match.group(1) or match.group(2) or ""
    title = re.sub(r"\\[A-Za-z]+\s*", "", title)
    title = re.sub(r"[{}]", "", title)
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def fields(block: str) -> set[str]:
    return {
        match.group(1).lower()
        for match in re.finditer(r"(?m)^\s*([A-Za-z][\w-]*)\s*=", block)
    }


def normalized_first_author(block: str) -> str:
    match = re.search(
        r"(?ims)^\s*author\s*=\s*(?:\{(.*?)\}|\"(.*?)\")\s*,?\s*$",
        block,
    )
    if not match:
        return ""
    author = (match.group(1) or match.group(2) or "").split(" and ", 1)[0]
    author = re.sub(r"[{}\\'\"~]", "", author).strip()
    family = author.split(",", 1)[0] if "," in author else author.split()[-1]
    return re.sub(r"[^a-z0-9]+", "", family.lower())


def canonical_score(key: str, entry_type: str, block: str, cited: set[str]) -> tuple[int, str]:
    entry_fields = fields(block)
    score = 0
    if key in cited:
        score += 10000
    conference_venue = bool(
        re.search(
            r"(?:booktitle|journal)\s*=\s*\{(?:NeurIPS|ICLR|ICML|CVPR|ICCV|ECCV|WACV|AAAI|FAccT)",
            block,
            flags=re.IGNORECASE,
        )
    )
    if entry_type == "inproceedings" or conference_venue:
        score += 350
    elif entry_type == "article":
        score += 250
    elif entry_type in {"book", "incollection"}:
        score += 150
    if "doi" in entry_fields:
        score += 120
    if "pages" in entry_fields:
        score += 50
    if "booktitle" in entry_fields or "journal" in entry_fields:
        score += 40
    if "author" in entry_fields:
        score += 30
    if "year" in entry_fields:
        score += 20
    if re.fullmatch(r"[a-z][a-z0-9]*\d{4}[a-z0-9]*", key):
        score += 80
    if re.search(r"DBLP:|[()/:]|^[A-Z]|-", key):
        score -= 60
    if re.search(r"arXiv preprint|\bCoRR\b", block, flags=re.IGNORECASE):
        score -= 80
    return score, key


def comment_alias(block: str, canonical_key: str) -> str:
    header = f"% DUPLICATE ALIAS; canonical key: {canonical_key}\n"
    lines = block.rstrip().splitlines()
    # BibTeX scans for entry markers even when ``@`` appears after ``%``.
    # Remove the marker from the stored alias header and restore it explicitly
    # in ``--restore`` mode.
    if lines and lines[0].startswith("@"):
        lines[0] = lines[0][1:]
    commented = "\n".join("% " + line if line else "%" for line in lines)
    return header + commented + "\n\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bib", type=Path)
    parser.add_argument("--tex-root", type=Path, required=True)
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()

    if args.restore:
        text = args.bib.read_text(encoding="utf-8")
        lines = text.splitlines()
        output: list[str] = []
        restoring = False
        restoring_header = False
        for line in lines:
            if line.startswith("% DUPLICATE ALIAS; canonical key:"):
                restoring = True
                restoring_header = True
                continue
            if restoring and line.startswith("% "):
                restored = line[2:]
                if restoring_header:
                    # Accept both the new BibTeX-safe format and aliases
                    # produced by older versions of this script.
                    if not restored.startswith("@"):
                        restored = "@" + restored
                    restoring_header = False
                output.append(restored)
                continue
            if restoring and line == "%":
                output.append("")
                continue
            restoring = False
            output.append(line)
        args.bib.write_text("\n".join(output) + "\n", encoding="utf-8")
        print("restored_duplicate_aliases")
        return

    cited: set[str] = set()
    for tex_path in args.tex_root.rglob("*.tex"):
        if tex_path.name.startswith("iclr2027_conference"):
            continue
        text = tex_path.read_text(encoding="utf-8", errors="ignore")
        for match in re.finditer(r"\\cite\w*\s*(?:\[[^]]*\]\s*){0,2}\{([^}]*)\}", text):
            cited.update(key.strip() for key in match.group(1).split(","))

    text = args.bib.read_text(encoding="utf-8")
    starts = list(re.finditer(r"(?m)^@(\w+)\s*\{\s*([^,\s]+)\s*,", text))
    entries: list[dict[str, object]] = []
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        block = text[match.start() : end]
        entries.append(
            {
                "start": match.start(),
                "end": end,
                "type": match.group(1).lower(),
                "key": match.group(2),
                "block": block,
                "title": normalized_title(block),
                "first_author": normalized_first_author(block),
            }
        )

    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for entry in entries:
        if entry["title"] and entry["first_author"]:
            groups[(str(entry["title"]), str(entry["first_author"]))].append(entry)

    replacements: dict[int, str] = {}
    for group in groups.values():
        if len(group) < 2:
            continue
        canonical = max(
            group,
            key=lambda entry: canonical_score(
                str(entry["key"]), str(entry["type"]), str(entry["block"]), cited
            ),
        )
        canonical_key = str(canonical["key"])
        for entry in group:
            if entry is canonical:
                continue
            replacements[int(entry["start"])] = comment_alias(str(entry["block"]), canonical_key)

    output = []
    cursor = 0
    for entry in entries:
        start = int(entry["start"])
        end = int(entry["end"])
        output.append(text[cursor:start])
        output.append(replacements.get(start, text[start:end]))
        cursor = end
    output.append(text[cursor:])
    args.bib.write_text("".join(output), encoding="utf-8")
    print(f"commented_aliases={len(replacements)}")


if __name__ == "__main__":
    main()
