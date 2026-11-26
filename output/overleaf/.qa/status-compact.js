JSON.stringify({title:document.title,dialog:document.querySelector('[role=dialog]')?.innerText,tail:document.body.innerText.slice(-1200),inputs:document.querySelectorAll('input[type=file]').length})
