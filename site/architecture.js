const ink = "#172a25";
const line = "#748c81";
const dark = "#103f35";
const soft = "#dcece4";
let sequence = 0;

function node(x, y, width, height, lines, accent = false) {
  const fill = accent ? dark : "#fff";
  const text = accent ? "#fff" : ink;
  const center = x + width / 2;
  const gap = 19;
  const first = y + height / 2 - (lines.length - 1) * gap / 2;
  return `<g><rect x="${x}" y="${y}" width="${width}" height="${height}" rx="7"
    fill="${fill}" stroke="${accent ? dark : "#bacbc2"}" stroke-width="1.3"/>
    <text text-anchor="middle" dominant-baseline="central" fill="${text}">
    ${lines.map((label, i) => `<tspan x="${center}" y="${first + gap * i}">${label}</tspan>`).join("")}
    </text></g>`;
}

function arrow(path, marker) {
  return `<path d="${path}" fill="none" stroke="${line}" stroke-width="1.5"
    stroke-linejoin="round" stroke-linecap="round" marker-end="url(#${marker})"/>`;
}

function phase(x, y, width, height, lines) {
  return node(x, y, width, height, lines).replace('fill="#fff"', `fill="${soft}"`);
}

function diagram(id, compact) {
  const marker = `${id}-arrow`;
  const diagramBody = compact
    ? [
      arrow("M85 60 V77 H63 V94", marker),
      arrow("M85 60 V77 H179 V94", marker),
      arrow("M275 60 V77 H296 V94", marker),
      arrow("M63 151 V174 H140 V193", marker),
      arrow("M179 151 V193", marker),
      arrow("M296 151 V174 H220 V193", marker),
      arrow("M179 245 V279", marker),
      arrow("M179 331 V365", marker),
      arrow("M98 518 H123", marker),
      arrow("M229 518 H253", marker),
      arrow("M302 487 H348 V218 H278", marker),
      node(8, 8, 154, 52, ["Market prices", "and factors"]),
      node(196, 8, 156, 52, ["Financial text"]),
      node(8, 95, 110, 56, ["Rolling", "HMM"]),
      node(124, 95, 110, 56, ["Market", "experts"]),
      node(240, 95, 112, 56, ["Text", "classifiers"]),
      node(83, 194, 193, 51, ["Mixture-of-experts", "controller"], true),
      node(83, 280, 193, 51, ["Trading rule"]),
      node(83, 366, 193, 51, ["Risk limits and", "simulated fills"]),
      phase(8, 488, 90, 60, ["Research", "agent"]),
      phase(124, 488, 105, 60, ["Candidate", "validation"]),
      phase(254, 488, 97, 60, ["Next fold’s", "gate"]),
    ].join("")
    : [
      arrow("M156 106 H174 V43 H196", marker),
      arrow("M156 106 H174 V122 H196", marker),
      arrow("M156 203 H196", marker),
      arrow("M358 43 H382 V99 H408", marker),
      arrow("M358 122 H408", marker),
      arrow("M358 203 H382 V145 H408", marker),
      arrow("M578 122 H616", marker),
      arrow("M766 122 H804", marker),
      arrow("M359 300 H409", marker),
      arrow("M577 300 H616", marker),
      arrow("M691 271 V234 H494 V159", marker),
      node(5, 78, 151, 56, ["Market prices", "and factors"]),
      node(5, 175, 151, 56, ["Financial text"]),
      node(198, 17, 160, 52, ["Rolling HMM"]),
      node(198, 96, 160, 52, ["Market experts"]),
      node(198, 177, 160, 52, ["Text classifiers"]),
      node(410, 87, 168, 70, ["Mixture-of-experts", "controller"], true),
      node(618, 94, 148, 56, ["Trading rule"]),
      node(806, 94, 164, 56, ["Risk limits and", "simulated fills"]),
      phase(198, 272, 161, 56, ["Research agent"]),
      phase(411, 272, 166, 56, ["Candidate", "validation"]),
      phase(618, 272, 148, 56, ["Next fold’s gate"]),
    ].join("");
  return `<svg viewBox="0 0 ${compact ? "360 558" : "978 337"}"
      class="arch-${compact ? "compact" : "wide"}" role="img" aria-labelledby="${id}-title ${id}-desc"
      xmlns="http://www.w3.org/2000/svg">
    <title id="${id}-title">Adaptive ensemble and simulated portfolio</title>
    <desc id="${id}-desc">Market data feeds the rolling hidden Markov model and market experts.
      Text classifiers supply additional forecasts. The controller weights the experts. A fixed policy
      determines positions, and the simulator applies risk limits and fills orders. A hosted agent
      proposes a new gate setting for a later fold, subject to a separate validation decision.</desc>
    <defs><marker id="${marker}" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6"
      orient="auto"><path d="M1 1 L7 4 L1 7" fill="none" stroke="${line}" stroke-width="1.4"/></marker></defs>
    ${diagramBody}</svg>`;
}

const styles = `
.ceci-architecture{container-type:inline-size;color:${ink};font-family:inherit;min-width:0}
.ceci-architecture svg{display:block;width:100%;height:auto;font-family:inherit;font-size:15px;font-weight:550}
.ceci-architecture .arch-compact{display:none}
.ceci-architecture table{border-collapse:collapse;table-layout:auto;width:100%;font:inherit;font-size:.9375rem;color:inherit}
.ceci-architecture th,.ceci-architecture td{border:0;border-top:1px solid #e4e7e4;padding:.75rem .4rem;
  vertical-align:top;text-align:left;line-height:1.45;background:transparent;white-space:normal}
.ceci-architecture th{font-weight:570;width:10.5rem;padding-left:0}
.ceci-architecture td{font-variant-numeric:tabular-nums}
.ceci-architecture .arch-state{display:inline-block;margin-right:1.1rem}
.ceci-architecture .arch-state:last-child{margin-right:0}
@container (max-width:720px){
  .ceci-architecture .arch-wide{display:none}
  .ceci-architecture .arch-compact{display:block;max-width:400px;margin:auto}
  .ceci-architecture th{width:7.1rem}
}
`;

function row(table, label, value) {
  const tr = table.insertRow();
  const th = document.createElement("th");
  th.scope = "row";
  th.textContent = label;
  tr.append(th);
  const td = tr.insertCell();
  if (typeof value === "string") td.textContent = value;
  else td.append(value);
}

export function renderArchitecture(el, study, sim, dayIndex) {
  const id = `ceci-architecture-${++sequence}`;
  const date = study?.dates?.[dayIndex] ?? sim?.dates?.[dayIndex];
  const index = sim?.dates?.indexOf(date) ?? -1;
  const states = sim?.posterior?.[index];
  const year = date ? Number(date.slice(0, 4)) : null;
  const review = study?.reviews?.find(r => r.year === year && r.method === "LLM review");
  const layout = document.createElement("div");
  layout.className = "ceci-architecture";
  layout.innerHTML = `<style>${styles}</style>${diagram(id+"-desktop", false)}${diagram(id+"-phone", true)}`;
  const table = document.createElement("table");
  table.setAttribute("aria-label", `Portfolio policy and research review${date ? " for " + date : ""}`);
  const body = table.createTBody();
  if (Array.isArray(states) && states.length === 3) {
    const group = document.createDocumentFragment();
    ["Bear", "Middle", "Bull"].forEach((name, i) => {
      const part = document.createElement("span");
      part.className = "arch-state";
      part.textContent = `${name} ${(100 * states[i]).toFixed(1)}%`;
      group.append(part);
    });
    row(body, `State on ${date}`, group);
  } else {
    row(body, "Market state", "No five-session state record for the selected date.");
  }
  row(body, "Trading rule (fixed policy)", "Subtract the chance of a fall from the chance of a rise. Keep up to five instruments with the highest scores above five percentage points. Give less volatile instruments larger targets, up to 15% of the account each. At the next opening price, buy or sell to reach the targets. Sell holdings that fall out of the selected group and keep spare money in cash.");
  row(body, "Risk-managed simulation", "A regime change raises the entry threshold to 0.15. A trailing price decline over 10% or daily volatility above 3.5% halves the position cap. The simulator limits designated exposure groups to 30%, uses a trailing stop of 2.5 average true ranges, and caps fills at 1% of the previous session’s dollar volume. A 10% portfolio drawdown triggers a five-session pause. Fills at the next open cost five basis points per side.");
  if (review) {
    const outcome = review.accepted
      ? `The candidate passed. The simulated gate used ${review.next_l2} in ${year}.`
      : `The incumbent setting of ${review.next_l2} remained in use for ${year}.`;
    row(body, `${year} research review`, `The hosted agent proposed L2 regularization of ${review.proposed_l2}. Three earlier validation windows supplied ${review.trades} simulated executions. ${outcome}`);
  }
  layout.append(table);
  el.replaceChildren(layout);
}
