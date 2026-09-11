#!/usr/bin/env python3
"""Audit BibTeX metadata against arXiv and Crossref without rewriting entries."""

from __future__ import annotations

import argparse
import concurrent.futures
import difflib
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests


def clean_latex(value: str) -> str:
    value = re.sub(r"\\[A-Za-z]+\s*", "", value)
    value = value.replace("{", "").replace("}", "")
    value = re.sub(r"[^A-Za-z0-9]+", " ", value).lower()
    return re.sub(r"\s+", " ", value).strip()


def similarity(left: str, right: str) -> float:
    return difflib.SequenceMatcher(None, clean_latex(left), clean_latex(right)).ratio()


def parse_entries(text: str) -> list[dict[str, str]]:
    starts = list(re.finditer(r"(?m)^@(\w+)\s*\{\s*([^,\s]+)\s*,", text))
    entries = []
    for index, match in enumerate(starts):
        block = text[match.start() : starts[index + 1].start() if index + 1 < len(starts) else len(text)]
        fields: dict[str, str] = {}
        for field in ("title", "author", "year", "doi", "eprint", "url", "journal", "booktitle"):
            found = re.search(
                rf"(?ims)^\s*{field}\s*=\s*(?:\{{(.*?)\}}|\"(.*?)\")\s*,?\s*$",
                block,
            )
            if found:
                fields[field] = (found.group(1) or found.group(2) or "").strip()
        entries.append({"key": match.group(2), "type": match.group(1).lower(), **fields})
    return entries


def arxiv_check(entry: dict[str, str]) -> dict[str, object] | None:
    arxiv_id = entry.get("eprint", "")
    if not re.fullmatch(r"\d{4}\.\d{4,5}(?:v\d+)?", arxiv_id):
        journal = entry.get("journal", "")
        match = re.search(r"arXiv:\s*(\d{4}\.\d{4,5})", journal, re.IGNORECASE)
        arxiv_id = match.group(1) if match else ""
    if not arxiv_id:
        return None
    response = requests.get(
        "https://export.arxiv.org/api/query",
        params={"id_list": arxiv_id},
        timeout=20,
    )
    response.raise_for_status()
    root = ET.fromstring(response.text)
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    item = root.find("atom:entry", namespace)
    if item is None:
        return {"source": "arxiv", "status": "not_found", "identifier": arxiv_id}
    title = " ".join((item.findtext("atom:title", default="", namespaces=namespace)).split())
    score = similarity(entry.get("title", ""), title)
    return {"source": "arxiv", "status": "verified" if score >= 0.9 else "mismatch", "score": score, "matched_title": title, "identifier": arxiv_id}


def crossref_check(entry: dict[str, str]) -> dict[str, object]:
    title = entry.get("title", "")
    if not title:
        return {"source": "crossref", "status": "missing_title"}
    params = {"query.title": clean_latex(title), "rows": 1, "select": "title,author,published,DOI,container-title"}
    response = requests.get(
        "https://api.crossref.org/works",
        params=params,
        headers={"User-Agent": "SafeFixBibliographyAudit/1.0 (mailto:oxx220000@utdallas.edu)"},
        timeout=25,
    )
    if response.status_code == 429:
        time.sleep(2)
        response = requests.get("https://api.crossref.org/works", params=params, timeout=25)
    response.raise_for_status()
    items = response.json()["message"]["items"]
    if not items:
        return {"source": "crossref", "status": "not_found"}
    item = items[0]
    matched_title = (item.get("title") or [""])[0]
    score = similarity(title, matched_title)
    status = "verified" if score >= 0.94 else "possible" if score >= 0.82 else "mismatch"
    return {
        "source": "crossref",
        "status": status,
        "score": score,
        "matched_title": matched_title,
        "doi": item.get("DOI", ""),
        "container_title": (item.get("container-title") or [""])[0],
    }


def doi_check(entry: dict[str, str]) -> dict[str, object] | None:
    doi = entry.get("doi", "").strip()
    if not doi:
        return None
    response = requests.get(
        "https://api.crossref.org/works/" + requests.utils.quote(doi, safe=""),
        headers={"User-Agent": "SafeFixBibliographyAudit/1.0 (mailto:oxx220000@utdallas.edu)"},
        timeout=20,
    )
    if response.status_code == 404:
        return {"source": "doi", "status": "not_found", "identifier": doi}
    response.raise_for_status()
    item = response.json()["message"]
    matched_title = (item.get("title") or [""])[0]
    score = similarity(entry.get("title", ""), matched_title)
    return {"source": "doi", "status": "verified" if score >= 0.9 else "mismatch", "score": score, "matched_title": matched_title, "identifier": doi}


def check_entry(entry: dict[str, str]) -> dict[str, object]:
    try:
        identifier_result = arxiv_check(entry)
        if identifier_result is None:
            identifier_result = doi_check(entry)
        result = identifier_result if identifier_result is not None else crossref_check(entry)
    except Exception as error:  # keep the audit running after transient API failures
        result = {"source": "network", "status": "error", "error": str(error)}
    return {"key": entry["key"], "title": entry.get("title", ""), **result}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bib", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--identifiers-only", action="store_true")
    args = parser.parse_args()

    entries = parse_entries(args.bib.read_text(encoding="utf-8"))
    if args.identifiers_only:
        entries = [entry for entry in entries if entry.get("eprint") or entry.get("doi") or "arxiv:" in entry.get("journal", "").lower()]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(check_entry, entries))

    counts: dict[str, int] = {}
    for result in results:
        status = str(result["status"])
        counts[status] = counts.get(status, 0) + 1
    payload = {"entry_count": len(entries), "status_counts": counts, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"entry_count": len(entries), "status_counts": counts}, indent=2))


if __name__ == "__main__":
    main()
