#!/usr/bin/env bash
set -u

cd "$(dirname "$0")"

while true; do
  find . -type f \
    \( -name '*.qmd' -o -name '*.js' -o -name '*.css' -o -name '*.scss' \
       -o -name '*.bib' -o -name '*.yml' -o -name '*.yaml' \
       -o -path './media/*' -o -path './slides/*' -o -path './manim-figures/*' \) \
    -not -path './main_files/*' -not -path './.quarto/*' \
    | entr -d quarto render main.qmd
done
