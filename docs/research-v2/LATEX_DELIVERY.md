# Native LaTeX / Overleaf delivery — 28 September 2026

**Later protocol amendment:** the same Overleaf project now contains the
five-session revision described in `docs/research-week1/STATUS.md`.
Its versioned source is `output/overleaf/week1/`, with PDF
`output/pdf/ML_TradeEngine_Research_Week1.pdf`. The files described below preserve
the original 63-session edition; its collection schedule is superseded.

The research paper is installed in the user's existing **ML_TradeEngine**
Overleaf project: https://www.overleaf.com/project/6aba2fa5d6050f063aaea85a.
The original starter authorship, Yash Kataria, is retained. Its starter source
is backed up in `output/overleaf/original-overleaf-main.tex`.

The delivered Overleaf build has 27 pages, an automatic contents page, 22
editable tables, 13 native TikZ/PGFPlots figures, three numbered equations,
14 BibTeX references and three appendices. The bibliography starts on its own
page (printed page 22; physical PDF page 24). Prose, tables, equations and
figures are native LaTeX. The PDF contains zero raster images.

Overleaf's final compile panel reports **0 errors, 0 warnings, 0 information
messages**. All 27 downloaded PDF pages were rendered and visually inspected.
The downloaded project source matches every entry in the local SHA-256
manifest. See `output/overleaf/VERIFICATION.json` for exact checks and hashes.

## Files

- `output/pdf/ML_TradeEngine_Research_LaTeX.pdf`: exact final Overleaf PDF.
- `output/overleaf/ML_TradeEngine_Research_Overleaf.zip`: portable source backup.
- `output/overleaf/ML_TradeEngine_Overleaf_Download.zip`: verified download of
  the remote project source.
- `output/overleaf/overleaf-upload/main.tex`: complete editable paper.
- `output/overleaf/overleaf-upload/references.bib`: editable bibliography.
- `output/overleaf/ml-trade-engine-research-latex/`: modular source and
  frozen statistics, claim map, citation evidence and conversion provenance.
- `scripts/latex/build_project.py` and `pack_project.py`: reproducible local
  conversion and packaging scripts. Local Tectonic/XeTeX pagination is 28
  pages; the authoritative delivered Overleaf pdfLaTeX pagination is 27.

## Formatting decisions

This is a long single-column research report, since a publication venue has
not been selected. It uses A4 paper, 11-point Latin Modern type, 25 mm margins,
numbered sections, linked citations and cross-references, table captions above
tables, figure captions below figures, right-aligned numeric columns and
minimal horizontal table rules. Short tables remain whole; the longer
evidence inventory repeats its header over page breaks. The requested contents
page is automatic. References and the reproduction appendix start on new pages.

Guidance consulted:

- [IEEE author templates](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/authoring-tools-and-templates/tools-for-ieee-authors/ieee-article-templates/): apply the eventual journal's required template at submission.
- [Overleaf contents](https://www.overleaf.com/learn/latex/Table_of_contents).
- [Overleaf bibliography management](https://www.overleaf.com/learn/latex/Bibliography_management_with_natbib).
- [Overleaf figure and table positioning](https://www.overleaf.com/learn/latex/Positioning_images_and_tables).
- [Overleaf tables](https://www.overleaf.com/learn/latex/Tables).

## Scientific scope

This delivery changes formatting and bibliographic structure, not experiments,
results or readiness gates. The conversion re-verifies the source hashes bound
to the frozen statistics. The prior manuscript, PDF, source evidence and sealed
releases remain preserved. Negative and inconclusive findings remain explicit.
Prospective sections remain interim; the scheduled collector and its external
order prohibition are unchanged.
