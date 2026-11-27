import {modelFlow} from './model-flow.js';

export function modelMap(root, guide, head) {
  root.innerHTML = head('Models, inputs and outputs', 'Each forecaster estimates the chance of a rise, little change or a fall.')+
    modelFlow+
    '<div class="model-branches"><figure><figcaption>Where market conditions enter</figcaption><p>S&amp;P 500 returns and volatility <span aria-hidden="true">→</span> Market-state model <span aria-hidden="true">→</span> Forecast controller</p></figure>'+
    '<figure><figcaption>Where reinforcement learning enters</figcaption><p><a class="link" href="#controller/router">Model router</a> picks the forecast <span aria-hidden="true">·</span> Portfolio allocator picks the investment in a separate test</p></figure></div>'+
    '<details class="protocol bottom"><summary>How the price models read the data</summary><div class="model-branches model-reading">'+
    '<figure><figcaption>Logistic regression</figcaption><p>Weighted inputs <span aria-hidden="true">→</span> Add them up <span aria-hidden="true">→</span> Three probabilities</p></figure>'+
    '<figure><figcaption>XGBoost and LightGBM</figcaption><p>Input values <span aria-hidden="true">→</span> Questions with thresholds <span aria-hidden="true">→</span> Combine tree outputs</p></figure>'+
    '<figure><figcaption>Four models that read price history</figcaption><p>20 sessions in order <span aria-hidden="true">→</span> LSTM and GRU remember earlier days, the convolutional model looks for nearby patterns, and the Patch Transformer compares blocks of five days.</p></figure>'+
    '</div></details>'+
    '<div class="model-map"><details class="protocol"><summary>Inputs and outputs for all 13 forecasts</summary><dl></dl></details><details class="protocol"><summary>Inputs and outputs for controllers and trading rules</summary><dl></dl></details></div>'+
    '<p class="note"><a class="link" href="#comparison">Compare on shared dates</a> · <a class="link" href="assets/ensemble-study.pdf#page=4">Model diagram in the PDF</a></p>';
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
