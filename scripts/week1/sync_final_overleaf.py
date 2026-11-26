from pathlib import Path
import json,time,zipfile
from overleaf_bridge import evaluate,download,PROJECT
out=Path('output/overleaf/week1')
old_url=json.loads(evaluate("JSON.stringify(Array.from(document.querySelectorAll('a[href]')).find(a=>a.href.includes('output.pdf')).href)"))
evaluate("Array.from(document.querySelectorAll('button')).find(b=>b.innerText.trim()==='upload\\nUpload').click();'opened'")
items=[{'name':name,'content':(out/name).read_text()} for name in ['main.tex','SOURCE_MANIFEST.json']]
code="(()=>{const i=document.querySelector('input[type=file]:not([webkitdirectory])');const d=new DataTransfer();for(const f of "+json.dumps(items)+")d.items.add(new File([f.content],f.name,{type:'text/plain'}));i.files=d.files;i.dispatchEvent(new Event('change',{bubbles:true}));return 'files selected'})()"
print(evaluate(code),flush=True)
for _ in range(40):
    time.sleep(.25)
    r=evaluate("(()=>{const b=Array.from(document.querySelectorAll('[role=dialog] button')).find(b=>b.innerText.trim()==='Overwrite');if(b){b.click();return 'overwritten'}return 'waiting'})()")
    if r=='overwritten':break
else:raise TimeoutError('Replacement dialog')
time.sleep(2)
evaluate("document.querySelector('[role=treeitem][aria-label=\"main.tex\"] .entity').click();'main'")
evaluate("(()=>{const b=Array.from(document.querySelectorAll('button')).find(b=>b.innerText.trim()==='Recompile');if(b)b.click();return 'compiling'})()")
for _ in range(80):
    time.sleep(.5)
    v=json.loads(evaluate("JSON.stringify({url:Array.from(document.querySelectorAll('a[href]')).find(a=>a.href.includes('output.pdf'))?.href,compiling:!Array.from(document.querySelectorAll('button')).some(b=>b.innerText.trim()==='Recompile')})"))
    if v.get('url') and v['url']!=old_url and not v['compiling']:break
else:raise TimeoutError('New compiled PDF')
download(PROJECT+'/download/zip?verify=week1-render-final','output/overleaf/ML_TradeEngine_Week1_Overleaf_Download.zip')
with zipfile.ZipFile('output/overleaf/ML_TradeEngine_Week1_Overleaf_Download.zip') as z:
    for p in out.iterdir():assert z.read(p.name)==p.read_bytes(),p.name
download(v['url'],'output/pdf/ML_TradeEngine_Research_Week1.pdf')
print('Final source archive verified and fresh compiled PDF downloaded.',flush=True)
