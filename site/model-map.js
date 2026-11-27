export function modelMap(root, guide, head) {
  root.innerHTML = head('What each model reads and returns', 'The forecasts supply probabilities. The controller combines or chooses them. The trading rule turns the result into positions.') +
    '<p class="note">The adapter and model-router comparison uses the same 480 trading sessions in 2024–2025. The four-year stress study uses 959 sessions in 2022–2025. Training data and model inputs can differ. Compare results shows the shared evaluation dates for each selected study.</p>' +
    '<div class="model-map"><section><h2>The 13 forecasts</h2><dl></dl></section><section><h2>From forecasts to trades</h2><dl></dl></section></div>' +
    '<p class="note"><a class="link" href="#controller">See a recorded forecast and its votes</a> · <a class="link" href="#comparison">Compare the models on shared dates</a> · <a class="link" href="assets/ensemble-study.pdf#page=3">Model map in the PDF</a></p>';
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
