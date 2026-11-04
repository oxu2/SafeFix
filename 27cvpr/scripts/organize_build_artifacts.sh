#!/usr/bin/env bash
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
paper_dir=$(dirname -- "$script_dir")
artifact_dir="$paper_dir/build_artifacts"

mkdir -p -- "$artifact_dir"
shopt -s nullglob

moved=0
for source_path in "$paper_dir"/AttriSelect.{aux,bbl,blg,brf,fdb_latexmk,fls,log,out,ptc,synctex.gz}; do
    [[ -f "$source_path" ]] || continue
    filename=${source_path##*/}
    if [[ -e "$artifact_dir/$filename" ]]; then
        printf 'Keeping %s: build_artifacts/%s already exists.\n' "$filename" "$filename" >&2
        continue
    fi
    mv -- "$source_path" "$artifact_dir/$filename"
    printf 'Moved %s to build_artifacts/%s\n' "$filename" "$filename"
    moved=$((moved + 1))
done

if ((moved == 0)); then
    printf 'No build artifacts to move.\n'
fi
