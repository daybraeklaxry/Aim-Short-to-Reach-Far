const cards = [...document.querySelectorAll('.demo-card')];
const videos = [...document.querySelectorAll('.paired-demo')];
const methods = [...document.querySelectorAll('[data-method-filter]')];
const tasks = [...document.querySelectorAll('[data-task-filter]')];
const motion = document.querySelector('#motion-toggle');
const status = document.querySelector('#gallery-status');
const description = document.querySelector('#method-description');
const visible = new Set();
const descriptions = {
  rank: 'AP-rank predicts how recorded actions will behave from the current state. These examples compare it with Direct after the start is displaced.',
  cem: 'AP-CEM searches for new actions toward an observed target. These examples compare it with final-goal CEM from the same starts, using the same frozen model and search settings.'
};
let method = 'rank';
let task = 'all';
let playing = !matchMedia('(prefers-reduced-motion: reduce)').matches;

function playback() {
  motion.textContent = playing ? 'Pause videos' : 'Play videos';
  motion.setAttribute('aria-pressed', String(!playing));
  for (const video of videos) {
    if (playing && visible.has(video) && !video.closest('.demo-card').hidden && !document.hidden) {
      video.muted = true;
      video.play().catch(() => {});
    } else video.pause();
  }
}

function filter() {
  methods.forEach(button => {
    const selected = button.dataset.methodFilter === method;
    button.classList.toggle('selected', selected);
    button.setAttribute('aria-pressed', String(selected));
  });
  tasks.forEach(button => {
    const value = button.dataset.taskFilter;
    button.hidden = value !== 'all' && !cards.some(card => card.dataset.method === method && card.dataset.task === value);
    const selected = value === task;
    button.classList.toggle('selected', selected);
    button.setAttribute('aria-pressed', String(selected));
  });
  cards.forEach(card => { card.hidden = card.dataset.method !== method || (task !== 'all' && card.dataset.task !== task); });
  description.textContent = descriptions[method];
  status.textContent = `${cards.filter(card => !card.hidden).length} ${method === 'rank' ? 'AP-rank' : 'AP-CEM'} demonstrations`;
  playback();
}

methods.forEach(button => button.addEventListener('click', () => {
  method = button.dataset.methodFilter;
  task = 'all';
  filter();
}));
tasks.forEach(button => button.addEventListener('click', () => { task = button.dataset.taskFilter; filter(); }));
motion.addEventListener('click', () => { playing = !playing; playback(); });
const observer = new IntersectionObserver(entries => {
  entries.forEach(entry => entry.isIntersecting ? visible.add(entry.target) : visible.delete(entry.target));
  playback();
}, {threshold:0.3});
videos.forEach(video => observer.observe(video));
document.addEventListener('visibilitychange', playback);
filter();
