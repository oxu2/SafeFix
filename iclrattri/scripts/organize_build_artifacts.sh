#!/usr/bin/env bash

set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
paper_dir=$(dirname -- "$script_dir")
artifact_dir="$paper_dir/build_artifacts"

mkdir -p -- "$artifact_dir"
shopt -s nullglob

moved=0
for source_path in "$paper_dir"/main.*; do
    filename=${source_path##*/}
    case "$filename" in
        main.tex | main.pdf)
            continue
            ;;
    esac

    mv -- "$source_path" "$artifact_dir/$filename"
    printf 'Moved %s to build_artifacts/%s\n' "$filename" "$filename"
    moved=$((moved + 1))
done

if ((moved == 0)); then
    printf 'No main build artifacts to move.\n'
fi
