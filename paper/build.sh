#!/usr/bin/env bash
# Keep auxiliary files in a temporary directory and place the PDF beside Main.tex.
set -euo pipefail
cd -- "$(dirname -- "$0")"
paper_build_dir=$(mktemp -d /tmp/network-interdiction-paper.XXXXXX)
echo "Build logs: $paper_build_dir"
export TEXINPUTS="$PWD:${TEXINPUTS:-}"
export BIBINPUTS="$PWD:${BIBINPUTS:-}"
export BSTINPUTS="$PWD:${BSTINPUTS:-}"
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$paper_build_dir" Main.tex > "$paper_build_dir/pass1.log"
(cd "$paper_build_dir" && bibtex Main > bibtex-build.log)
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$paper_build_dir" Main.tex > "$paper_build_dir/pass2.log"
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$paper_build_dir" Main.tex > "$paper_build_dir/pass3.log"
cp "$paper_build_dir/Main.pdf" Main.pdf
echo "PDF: $PWD/Main.pdf"
