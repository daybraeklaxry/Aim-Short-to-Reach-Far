const study = JSON.parse(document.querySelector('#results-data').textContent);
const taskNames = {cube:'Cube',pusht:'PushT',reacher:'Reacher',tworoom:'TwoRoom'};
let resultRule = 'cem';
let resultStart = 'Standard';
let ablationStart = 'Standard';
const percent = value => value.toFixed(1);

function selectButton(attribute, value) {
  document.querySelectorAll(`[${attribute}]`).forEach(button => button.setAttribute('aria-pressed', String(button.getAttribute(attribute) === value)));
}
function renderMainResults() {
  const columns = resultRule === 'cem'
    ? [['AP-CEM','AP-CEM'],['CEM (final goal)','Final goal'],['CEM (learned target)','Learned target'],['LeWM planner','LeWM']]
    : [['AP-rank','AP-rank'],['Rank (final goal)','Final goal'],['Rank (learned target)','Learned target'],['Direct','Direct']];
  const table = document.querySelector('#main-results-table');
  table.querySelector('thead').innerHTML = '<tr><th scope="col">Task</th>'+columns.map(([arm,label],i)=>`<th scope="col"${i===0?' class="ap-column"':''}>${label}</th>`).join('')+'</tr>';
  const cells = values => values.map((value,i)=>`<td${i===0?' class="ap-column"':''}>${percent(value)}</td>`).join('');
  const rows = Object.entries(taskNames).map(([task,label])=>`<tr><th scope="row">${label}</th>${cells(columns.map(([arm])=>study.main[resultStart][task][arm]))}</tr>`);
  const means = columns.map(([arm])=>Object.keys(taskNames).reduce((sum,task)=>sum+study.main[resultStart][task][arm],0)/4);
  rows.push(`<tr class="mean-row"><th scope="row">Mean</th>${cells(means)}</tr>`);
  table.querySelector('tbody').innerHTML = rows.join('');
  table.querySelector('caption').textContent = `${resultStart} starts. ${resultRule==='cem'?'CEM and LeWM':'Recorded-action ranking'}. Success in percent.`;
  document.querySelector('#results-description').textContent = resultRule === 'cem'
    ? 'Final-goal, learned-target, and AP-CEM use the same five-action search. The LeWM planner uses 25-action plans. Main-evaluation goal offsets range from 100 to 150 actions.'
    : 'The three ranking methods score the same eight retrieved action blocks with different targets. Direct executes the closest record’s action without prediction. Main-evaluation goal offsets range from 100 to 150 actions.';
}
document.querySelectorAll('[data-rule]').forEach(button=>button.addEventListener('click',()=>{resultRule=button.dataset.rule;selectButton('data-rule',resultRule);renderMainResults();}));
document.querySelectorAll('[data-start]').forEach(button=>button.addEventListener('click',()=>{resultStart=button.dataset.start;selectButton('data-start',resultStart);renderMainResults();}));
renderMainResults();

document.querySelector('#quality-rows').innerHTML = Object.entries(taskNames).map(([task,label])=>{
  const error=study.quality[task];
  const learned=study.main.Standard[task]['CEM (learned target)'];
  const observed=study.main.Standard[task]['AP-CEM'];
  return `<tr><th scope="row">${label}</th><td>${error.Learned.toFixed(3)}</td><td>${error.Observed.toFixed(3)}</td><td>${learned>observed?'<strong>':''}${percent(learned)}${learned>observed?'</strong>':''}</td><td>${observed>learned?'<strong>':''}${percent(observed)}${observed>learned?'</strong>':''}</td></tr>`;
}).join('');

function renderAblations() {
  const selected=study.ablations[ablationStart];
  for (const kind of ['offset','memory']) {
    const rows=selected[kind];
    const best={cem_observed:Math.max(...rows.map(row=>row.cem_observed)),rank_observed:Math.max(...rows.map(row=>row.rank_observed))};
    document.querySelector(`#${kind}-rows`).innerHTML=rows.map(row=>`<tr><th scope="row">${row.label}</th>${['cem_observed','rank_observed'].map(arm=>`<td>${row[arm]===best[arm]?'<strong>':''}${percent(row[arm])}${row[arm]===best[arm]?'</strong>':''}</td>`).join('')}</tr>`).join('');
  }
  document.querySelector('#retrieval-bars').innerHTML=selected.retrieval.map(row=>`<div class="bar-row"><div><span>${row.label}</span><strong>${percent(row.value)}%</strong></div><div class="bar-track" aria-hidden="true"><span style="width:${row.value}%"></span></div></div>`).join('');
  document.querySelector('#retrieval-bars').setAttribute('aria-label',`AP-rank mean success at ${ablationStart.toLowerCase()} starts`);
}
document.querySelectorAll('[data-ablation-start]').forEach(button=>button.addEventListener('click',()=>{ablationStart=button.dataset.ablationStart;selectButton('data-ablation-start',ablationStart);renderAblations();}));
renderAblations();
document.querySelector('#anchor-rows').innerHTML=['Standard','Perturbed'].map(start=>`<tr><th scope="row">${start}</th><td>${percent(study.ablations[start].anchor.transport)}</td><td><strong>${percent(study.ablations[start].anchor.observed)}</strong></td></tr>`).join('');

function chooseTarget(target) {
  document.querySelector('.target-explainer').dataset.scoring = target;
  selectButton('data-target',target);
  const coordinates=target==='final'?[545,205]:[330,65];
  const candidates=[[370,230],[285,145],[385,350]];
  const distances=candidates.map(([x,y])=>(x-coordinates[0])**2+(y-coordinates[1])**2);
  const selected=distances.indexOf(Math.min(...distances))+1;
  document.querySelectorAll('.action-path,.candidate-point,.score-line').forEach(element=>{
    const id=Number(element.dataset.path||element.dataset.point||element.dataset.line);
    element.classList.toggle('chosen',id===selected);
  });
  document.querySelectorAll('.score-line').forEach(line=>{line.setAttribute('x2',coordinates[0]);line.setAttribute('y2',coordinates[1]);});
  document.querySelector('#final-target-point').classList.toggle('active-target',target==='final');
  document.querySelector('#observed-target-point').classList.toggle('active-target',target==='observed');
  document.querySelector('#selected-action').textContent=`The planner selects u${selected===1?'₁':'₂'}.`;
  document.querySelector('#target-explanation').textContent=target==='final'
    ? 'Its predicted endpoint is closest to the final goal, so it receives the lowest cost.'
    : 'The endpoint of u₂ is farther from the final goal than u₁’s, but closer to the observed target on the recorded route.';
}
document.querySelectorAll('[data-target]').forEach(button=>button.addEventListener('click',()=>chooseTarget(button.dataset.target)));
chooseTarget('final');
