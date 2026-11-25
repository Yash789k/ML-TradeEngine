JSON.stringify(Array.from(document.querySelectorAll('[role=dialog]')).map(e=>({text:e.innerText,buttons:Array.from(e.querySelectorAll('button')).map(b=>b.innerText)})))
