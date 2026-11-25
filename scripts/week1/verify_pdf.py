"""Render every amended Overleaf page and check schedule/content invariants."""
from pathlib import Path
from pypdf import PdfReader
from PIL import Image,ImageDraw
import json,subprocess,hashlib
ROOT=Path(__file__).resolve().parents[2]
p=ROOT/'output/pdf/ML_TradeEngine_Research_Week1.pdf';doc=PdfReader(p)
out=ROOT/'data/research-week1/pdf-final-qa';out.mkdir(exist_ok=True)
texts=[page.extract_text() for page in doc.pages];text='\n'.join(texts)
assert 'five fixed XNYS' in text and '2026-10-14 04:30 HKT' in text
assert 'Observed-session milestone 63' not in text
assert 'Contents' in texts[1]
refs=[i+1 for i,t in enumerate(texts) if '\nReferences\n' in '\n'+t+'\n']
changes=[i+1 for i,t in enumerate(texts) if any(s in t for s in ['five fixed XNYS','Fifth and final decision','Registered five-session','NEW-DATED-RUN'])]
binary='/Users/a/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override/pdftoppm'
subprocess.run([binary,'-r','50','-png',str(p),str(out/'page')],check=True,capture_output=True)
for i in changes:subprocess.run([binary,'-f',str(i),'-l',str(i),'-r','110','-singlefile','-png',str(p),str(out/f'changed-{i:02}')],check=True,capture_output=True)
for first in range(0,len(doc.pages),9):
 sheet=Image.new('RGB',(1320,1875),'white')
 for n,i in enumerate(range(first,min(first+9,len(doc.pages)))):
  im=Image.open(out/f'page-{i+1:02}.png').convert('RGB');tile=Image.new('RGB',(440,625),'#dddddd');tile.paste(im,((440-im.width)//2,25));ImageDraw.Draw(tile).text((10,7),f'Page {i+1}',fill='black');sheet.paste(tile,((n%3)*440,(n//3)*625))
 sheet.save(out/f'contact-{first//9+1}.png')
qa={'pages':len(doc.pages),'changed_pages':changes,'references_pages':refs,'raster_images':sum(len(p.images) for p in doc.pages),'contents_page':2,'pdf_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
assert qa['raster_images']==0
(out/'text-checks.json').write_text(json.dumps(qa,indent=2));(out/'extracted.txt').write_text(text)
print(json.dumps(qa))
