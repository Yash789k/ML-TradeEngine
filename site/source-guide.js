const percent = value => (value * 100).toFixed(1) + '%';

export function sourceGuide(host, evidence) {
  const section = document.createElement('section');
  section.className = 'bottom source-guide';
  section.innerHTML = '<h2>Where the inputs come from</h2><dl></dl>';
  const populate = (list, entries) => {
    for (const [title, explanation] of entries) {
      const key = document.createElement('dt');
      const detail = document.createElement('dd');
      key.textContent = title;
      detail.textContent = explanation;
      list.append(key, detail);
    }
  };
  populate(section.querySelector('dl'), evidence.sources);
  section.insertAdjacentHTML('beforeend', '<h2>What OpenAI and Jev do</h2><dl></dl>');
  populate(section.lastElementChild, evidence.uses);
  const note = document.createElement('p');
  note.className = 'note';
  const apple = evidence.apple;
  note.textContent = 'Example from the recorded Apple decision on '+apple.decision+'. The '+apple.packet+
    ' asset digest gave Apple a '+percent(apple.rawUp)+' chance of rising. After calibration its forecast was '+percent(apple.calibratedUp)+
    ' up and it received '+percent(apple.weight)+' of the vote. The combined forecast used all 13 votes.';
  section.append(note);
  const link = document.createElement('a');
  link.className='link';
  link.href='#controller';
  link.textContent='See the combined Apple forecast';
  section.append(link);
  section.insertAdjacentHTML('beforeend', '<h2>Company text coverage</h2><p class="note">The chart counts sampled records for the ten companies with filings. The remaining 12 instruments are funds. They receive market inputs, including the Fed statement forecasts.</p><figure><label class="field" for="companySource">Count<select id="companySource"><option value="news">Headlines</option><option value="filing">Annual filings</option></select></label><div class="company-bars"></div></figure>');
  const chooser = section.querySelector('#companySource');
  const chart = section.querySelector('.company-bars');
  function update() {
    const kind = chooser.value;
    const max = Math.max(...evidence.companies.map(row => row[kind]));
    chart.replaceChildren();
    for (const company of evidence.companies) {
      const row = document.createElement('div');
      row.className = 'pipeline-bar';
      const name = document.createElement('span');
      const count = document.createElement('strong');
      const track = document.createElement('div');
      const bar = document.createElement('i');
      name.textContent = company.symbol;
      count.textContent = company[kind].toLocaleString();
      track.className = 'track';
      track.setAttribute('aria-hidden','true');
      bar.style.width = (100 * company[kind] / max) + '%';
      track.append(bar);
      row.append(name,count,track);
      chart.append(row);
    }
  }
  chooser.onchange = update;
  update();
  host.append(section);
}
