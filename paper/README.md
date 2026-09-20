# Paper Source

Compiled PDF: `cose.pdf` (latest build).

## Requirements

- `pdflatex` (TeX Live 2023 or newer)
- `bibtex`
- Elsevier `elsarticle` document class (included in TeX Live)
- All figure PDFs in `../figures/`

## Compile

```bash
pdflatex cose.tex
bibtex cose
pdflatex cose.tex
pdflatex cose.tex
```

Or via the project Makefile from the repository root:

```bash
make paper
```

## Structure

- `cose.tex` — main manuscript (title, authors, abstract, 6 sections, references)
- `references.bib` — BibTeX database with 41 entries
- `cover_letter.tex` — cover letter for JISA submission
- `cose.pdf` — compiled output (PDF/A compatible)

## Notes

- Section numbering uses Arabic numerals (Elsevier style), not Roman (IEEE style).
- Limitations and Future Work are subsections of the Discussion (5.5, 5.6), not the Conclusion.
- The `graphicspath` in `cose.tex` expects figures at `../figures/` and `../jpeg/`.
