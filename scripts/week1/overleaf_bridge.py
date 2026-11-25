"""Interact with the authorized existing Overleaf tab in the user's Chrome."""
import base64,json,subprocess,time
from pathlib import Path
PROJECT='https://www.overleaf.com/project/6aba2fa5d6050f063aaea85a'

def evaluate(code):
    command="var found; Application(646).windows().forEach(w=>w.tabs().forEach(t=>{if(t.url()==="+json.dumps(PROJECT)+")found=t})); if(!found)throw Error('Existing Overleaf tab missing'); found.execute({javascript:"+json.dumps(code)+"});"
    r=subprocess.run(['osascript','-l','JavaScript','-e',command],text=True,capture_output=True,check=True)
    return r.stdout.strip()

def download(url,path):
    evaluate("window.__week1download=null;fetch("+json.dumps(url)+",{credentials:'include'}).then(r=>{if(!r.ok)throw Error('Download '+r.status);return r.arrayBuffer()}).then(b=>{let a=new Uint8Array(b),s='';for(let i=0;i<a.length;i+=16384)s+=String.fromCharCode(...a.subarray(i,i+16384));window.__week1download={data:btoa(s)}}).catch(e=>window.__week1download={error:String(e)});'started'")
    for _ in range(100):
        time.sleep(.2)
        result=json.loads(evaluate('JSON.stringify(window.__week1download)'))
        if result is not None:
            if 'error' in result:raise RuntimeError(result['error'])
            Path(path).write_bytes(base64.b64decode(result['data']));return
    raise TimeoutError('Overleaf download')

if __name__=='__main__':
    download(PROJECT+'/download/zip','data/research-week1/overleaf-before-amendment.zip')
    print('Preserved current Overleaf source archive.')
