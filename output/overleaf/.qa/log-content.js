JSON.stringify({text:document.body.innerText.slice(-15000),links:Array.from(document.querySelectorAll('a')).filter(e=>/log|output/i.test(e.innerText)).map(e=>({text:e.innerText,href:e.href}))})
