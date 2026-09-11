#!/usr/bin/env python3
"""Normalize venue names and protected proper names in a BibTeX file."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


VENUE_RULES = (
    (r"journal of machine learning research|the journal of machine learning research", "JMLR"),
    (r"neural information processing systems|\bneurips\b|\bnips\b", "NeurIPS"),
    (r"international conference on learning representations|\biclr\b", "ICLR"),
    (r"international conference on machine learning|\bicml\b", "ICML"),
    (r"computer vision and pattern recognition.*workshop|\bcvpr\b.*workshop", "CVPR Workshops"),
    (r"computer vision and pattern recognition|\bcvpr\b", "CVPR"),
    (r"international conference on computer vision|\biccv\b", "ICCV"),
    (r"european conference on computer vision|computer vision.?[--]+eccv|\beccv\b", "ECCV"),
    (r"winter conference on applications of computer vision|\bwacv\b", "WACV"),
    (r"aaai conference on artificial intelligence|\baaai\b", "AAAI"),
    (r"fairness,? accountability,? and transparency|\bfacct\b|\bfat\*?\b", "FAccT"),
)

PROPER_NAMES = (
    "ImageNet",
    "COCO",
    "KITTI",
    "CelebA",
    "SafeFix",
    "AttriSelect",
    "HiBug2",
    "HiBug",
    "TALISMAN",
    "PPAL",
    "Qwen3-VL",
    "ControlNet",
    "InstructPix2Pix",
    "ICEdit",
    "ResNet",
    "BERT",
    "CLIP",
    "Stable Diffusion",
    "SDXL",
    "T2I-CompBench",
    "HRS-Bench",
    "GenEval",
    "TIFA",
    "Pick-a-Pic",
    "FACTS",
    "LAION-5B",
    "FLUX.1",
)


def normalize_venue(value: str) -> str:
    plain = re.sub(r"[{}]", "", value)
    plain = re.sub(r"\s+", " ", plain).strip()
    lowered = plain.lower()
    for pattern, replacement in VENUE_RULES:
        if re.search(pattern, lowered, flags=re.IGNORECASE):
            return replacement
    return value


def protect_proper_names(value: str) -> str:
    for name in PROPER_NAMES:
        pattern = re.compile(r"(?<![A-Za-z{])" + re.escape(name) + r"(?![A-Za-z}])", re.IGNORECASE)
        value = pattern.sub("{" + name + "}", value)
    return value


def transform_fields(text: str) -> str:
    field_pattern = re.compile(
        r"(?im)^(?P<indent>\s*)(?P<field>journal|booktitle|title)\s*=\s*"
        r"(?P<open>[{\"])(?P<value>.*?)(?P<close>[}\"])(?P<comma>,?)\s*$"
    )

    def replace(match: re.Match[str]) -> str:
        field = match.group("field").lower()
        value = match.group("value")
        if field in {"journal", "booktitle"}:
            value = normalize_venue(value)
        else:
            value = protect_proper_names(value)
        return (
            f"{match.group('indent')}{field}={match.group('open')}"
            f"{value}{match.group('close')}{match.group('comma')}"
        )

    text = field_pattern.sub(replace, text)
    text = re.sub(r"(?im)^@inproceedings\{", "@inproceedings{", text)
    text = re.sub(r"(?im)^@article\{", "@article{", text)
    text = re.sub(r"(?im)^@misc\{", "@misc{", text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bib", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    original = args.bib.read_text(encoding="utf-8")
    normalized = transform_fields(original)
    if args.check:
        raise SystemExit(0 if normalized == original else 1)
    args.bib.write_text(normalized, encoding="utf-8")


if __name__ == "__main__":
    main()
