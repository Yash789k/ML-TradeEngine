const methods = [
  ['HMM gate','Scheduled refit with the existing setting'],
  ['LLM review','Language-model proposal (OpenAI)'],
  ['Conventional review','Fixed proposal'],
  ['Random review','Random proposal'],
];

export function reviewResults(root, periods, studies, {head, table, pct}) {
  root.innerHTML = head('Did any proposed change get adopted?', 'Each year the gate refits on schedule. The review methods can propose changing one setting that controls how strongly the gate restrains its coefficients.') +
    '<div class="controls vote-controls"><label class="field" for="reviewHorizon">Forecast horizon<select id="reviewHorizon"><option value="five">Five trading sessions</option><option value="one">One trading session</option></select></label></div>'+
    '<div id="reviewYears"></div><div id="reviewReturns" class="bottom"></div>'+
    '<p class="note">The language models that supply forecasts are shared inputs. The OpenAI research agent can suggest a different gate setting. The archived experiment contains no retraining of the language forecasters by the agent.</p>'+
    '<p class="note"><a class="link" href="#comparison/pipeline">Compare the mixture and its controls</a> · <a class="link" href="assets/ensemble-study.pdf#page=9">Annual review in the report</a></p>';
  const selector = root.querySelector('#reviewHorizon');
  function update() {
    const group = periods.groups.find(group=>group.id===selector.value);
    const reviews = studies[selector.value].reviews;
    const years = [2023,2024,2025];
    root.querySelector('#reviewYears').innerHTML =
      '<h2>What the gate used in each evaluation year</h2>'+
      table(['Review method',...years],methods.map(([method,label])=>[
        label,...years.map(year=>{
          const review=reviews.find(row=>row.year===year&&row.method===method);
          if (!review) return 'Existing setting (1.0)';
          if(review.accepted) return 'Adopted '+review.next_l2.toFixed(1);
          if(review.proposed_l2===review.next_l2) return 'Kept '+review.next_l2.toFixed(1);
          return 'Kept '+review.next_l2.toFixed(1)+' after testing '+review.proposed_l2.toFixed(1);
        }),
      ]));
    root.querySelector('#reviewReturns').innerHTML =
      '<h2>Return after trading costs</h2>'+
      table(['Review method',...years,'2023–2025'],methods.map(([method,label])=>[
        label,...[...years.map(String),'all-2023-2025'].map(id=>{
          const period=group.periods.find(row=>row.id===id);
          return pct(period.rows.find(row=>row.model===method).net_return,2);
        }),
      ]));
  }
  selector.onchange=update;
  update();
}
