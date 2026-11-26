"""Pack an equivalent flat project for uploading into an existing Overleaf project."""
from pathlib import Path
import hashlib,json,re,shutil,zipfile

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'output/overleaf/ml-trade-engine-research-latex'
OUT=ROOT/'output/overleaf/overleaf-upload'
OUT.mkdir(parents=True,exist_ok=True)

def inline(text):
    def replace(m):
        rel=m[1]+('' if m[1].endswith('.tex') else '.tex')
        return '% Begin '+rel+'\n'+inline((SOURCE/rel).read_text())+'\n% End '+rel+'\n'
    return re.sub(r'\\input\{([^}]+)\}',replace,text)

main=inline((SOURCE/'main.tex').read_text()).replace('{data/portfolio-','{portfolio-')
(OUT/'main.tex').write_text(main)
for p in [SOURCE/'references.bib',SOURCE/'latexmkrc',*sorted((SOURCE/'data').glob('*.csv'))]:
    shutil.copy2(p,OUT/p.name)
(OUT/'README.md').write_text('''# ML-TradeEngine research paper

Author: Yash Kataria, retained from the existing Overleaf project.

Open `main.tex` and compile with pdfLaTeX. Overleaf runs BibTeX automatically.
All prose, 22 tables, three displayed equations and 13 figures are editable
LaTeX. Figures use TikZ/PGFPlots; no text or charts are raster images.
The portfolio CSV files contain the complete saved account histories used by
the three-panel portfolio chart. Bibliographic metadata is in references.bib.

Formatting: single-column A4, 11-point type, 25 mm margins, numbered sections,
automatic table of contents, linked numerical citations, table captions above
tables, figure captions below figures, repeated table headers, separate
bibliography page, lettered appendices, running heads and page numbers.

Evidence cutoff: 25 September 2026; report date: 28 September 2026.
Prospective sections remain interim. This formatting conversion does not
change empirical results or promote any readiness gate.

The modular source, original statistics, claim map, citation verification and
conversion provenance are retained locally under
output/overleaf/ml-trade-engine-research-latex/ in the research repository.

Formatting guidance consulted:
- https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/authoring-tools-and-templates/tools-for-ieee-authors/ieee-article-templates/
- https://www.overleaf.com/learn/latex/Table_of_contents
- https://www.overleaf.com/learn/latex/Bibliography_management_with_natbib
- https://www.overleaf.com/learn/latex/Positioning_images_and_tables
- https://www.overleaf.com/learn/latex/Tables

A specific journal has not been selected; its required template should take
precedence at submission. The requested contents page is appropriate to this
long research report and may be removed if the eventual journal requires it.
''')
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='SOURCE_MANIFEST.json'}
(OUT/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
dest=OUT.parent/'ML_TradeEngine_Research_Overleaf.zip'
with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as archive:
    for p in sorted(OUT.iterdir()):
        if p.is_file():archive.write(p,p.name)
print(json.dumps({'archive':str(dest),'files':len(manifest)+1,'bytes':dest.stat().st_size,'source_bytes':len(main.encode())}))
