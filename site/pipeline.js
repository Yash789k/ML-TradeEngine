export function pipelineView(root, guide, head) {
  root.innerHTML = head('From raw data to results', 'The original proposal asks whether changing whose forecast counts can improve returns after allowing for risk.') +
    '<div class="pipeline-layout"><ol class="pipeline-stages"></ol><aside class="pipeline-metrics"><h2>How to read the scores</h2><dl></dl><a class="link" href="#comparison">See the historical comparisons</a></aside></div>';
  const stages = root.querySelector('.pipeline-stages');
  for (const [title, explanation] of guide.stages) {
    const row = document.createElement('li');
    const heading = document.createElement('h2');
    const text = document.createElement('p');
    heading.textContent = title;
    text.textContent = explanation;
    row.append(heading, text);
    stages.append(row);
  }
  const metrics = root.querySelector('.pipeline-metrics dl');
  for (const [title, explanation] of guide.metrics) {
    const label = document.createElement('dt');
    const text = document.createElement('dd');
    label.textContent = title;
    text.textContent = explanation;
    metrics.append(label, text);
  }
}
