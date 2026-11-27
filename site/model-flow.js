export const modelFlow = `<ol class="model-flow" aria-label="From inputs to simulated trades">
  <li><div><a href="#data">Input data</a><ul><li>Prices and volume</li><li>Company text and Fed statements</li><li>Investment factors</li></ul></div></li>
  <li><div><a href="#models">Direction forecasts</a><ul><li>Seven price models</li><li>Five language forecasts</li><li>One factor model</li></ul></div></li>
  <li><div><a href="#controller">Forecast controller</a><ul><li>Blend forecasts using learned weights</li><li>Or select one model in the router experiment</li></ul></div></li>
  <li><div><a href="#simulation">Simulated trades</a><ul><li>Rank stocks and set position sizes</li><li>Apply costs at the next open</li></ul></div></li>
</ol>`;
