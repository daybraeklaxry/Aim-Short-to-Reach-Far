const cards = [...document.querySelectorAll('.demo-card')];
const videos = [...document.querySelectorAll('#demos video')];
const filters = [...document.querySelectorAll('.demo-filter')];
const toggle = document.querySelector('#autoplay-toggle');
const count = document.querySelector('#demo-count');
const visible = new Set();
let autoplay = !matchMedia('(prefers-reduced-motion: reduce)').matches;
function playback() {
  toggle.textContent = `Autoplay: ${autoplay ? 'on' : 'off'}`;
  toggle.setAttribute('aria-pressed', String(autoplay));
  for (const video of videos) {
    const hidden = video.closest('.demo-card')?.hidden;
    if (autoplay && visible.has(video) && !hidden && !document.hidden) {
      video.muted = true;
      video.play().catch(() => {});
    } else video.pause();
  }
}
const observer = new IntersectionObserver(entries => {
  for (const entry of entries) {
    if (entry.isIntersecting) visible.add(entry.target);
    else visible.delete(entry.target);
  }
  playback();
}, {threshold: 0.3});
videos.forEach(video => observer.observe(video));
toggle.addEventListener('click', () => { autoplay = !autoplay; playback(); });
filters.forEach(button => button.addEventListener('click', () => {
  const task = button.dataset.filter;
  filters.forEach(b => {
    const selected = b === button;
    b.classList.toggle('active', selected);
    b.setAttribute('aria-pressed', String(selected));
  });
  cards.forEach(card => { card.hidden = task !== 'all' && card.dataset.task !== task; });
  const n = cards.filter(card => !card.hidden).length;
  count.textContent = `${n} paired demonstrations${task === 'all' ? '' : ' · ' + button.textContent.replace('2', '').trim()}`;
  playback();
}));
document.addEventListener('visibilitychange', playback);
playback();
