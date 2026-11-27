import {modelFlow} from './model-flow.js';

export function modelMap(root, guide, head) {
  root.innerHTML = head('Models, inputs and outputs', 'Each forecaster estimates the chance of a rise, little change or a fall.')+
    modelFlow+
    '<div class="model-branches"><figure><figcaption>Where market conditions enter</figcaption><p>S&amp;P 500 returns and volatility <span aria-hidden="true">→</span> Market-state model <span aria-hidden="true">→</span> Forecast controller</p></figure>'+
    '<figure><figcaption>Where reinforcement learning enters</figcaption><p><a class="link" href="#controller/router">Model router</a> picks the forecast <span aria-hidden="true">·</span> Portfolio allocator picks the investment in a separate test</p></figure></div>'+
    '<div class="model-map"><details class="protocol"><summary>Inputs and outputs for all 13 forecasts</summary><dl></dl></details><details class="protocol"><summary>Inputs and outputs for controllers and trading rules</summary><dl></dl></details></div>'+
    '<p class="note"><a class="link" href="#comparison">Compare on shared dates</a> · <a class="link" href="assets/ensemble-study.pdf#page=3">Model diagram in the PDF</a></p>';
  const lists = root.querySelectorAll('.model-map dl');
  for (const [i, group] of [guide.forecast, guide.controller].entries()) {
    for (const [name, input, output] of group) {
      const term = document.createElement('dt');
      const detail = document.createElement('dd');
      term.textContent = name;
      for (const [label, value] of [['Reads', input], ['Returns', output]]) {
        const line = document.createElement('p');
        const key = document.createElement('strong');
        key.textContent = label + ' ';
        line.append(key, value);
        detail.append(line);
      }
      lists[i].append(term, detail);
    }
  }
}
