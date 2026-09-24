/* Paired views of the same recorded experiment, kept on a shared clock. */
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const pairs = [...document.querySelectorAll('.demo')].map(element => ({
  element,
  videos: [...element.querySelectorAll('video')],
  view: element.querySelector('.video-pair'),
  restart: element.querySelector('.pair-restart'),
  inView: false, userPaused: false, userStarted: false, loaded: false, starting: false,
}));

function wantsPlayback(pair) {
  const disclosure = pair.element.closest('details');
  return pair.inView && !document.hidden && (!disclosure || disclosure.open)
    && !pair.userPaused && (!reducedMotion.matches || pair.userStarted);
}
function preparePosters(pair) {
  pair.videos.forEach(video => { if (!video.poster) video.poster = video.dataset.poster; });
}
function loadPair(pair) {
  if (pair.loaded) return;
  pair.loaded = true;
  preparePosters(pair);
  pair.videos.forEach(video => {
    video.muted = true;
    video.src = video.dataset.src;
    video.load();
  });
}
function pausePair(pair) { pair.videos.forEach(video => video.pause()); }
async function updatePair(pair) {
  if (!wantsPlayback(pair)) { pausePair(pair); return; }
  loadPair(pair);
  if (pair.starting || pair.videos.some(video => video.readyState < 3)) return;
  if (pair.videos.every(video => !video.paused)) return;
  pair.starting = true;
  const [clock, follower] = pair.videos;
  if (Math.abs(clock.currentTime - follower.currentTime) > .08) follower.currentTime = clock.currentTime;
  try {
    await Promise.all(pair.videos.map(video => video.play()));
    if (!wantsPlayback(pair)) pausePair(pair);
  } catch {
    // If autoplay is blocked, Replay or a tap on either video starts the pair.
    pausePair(pair);
  } finally { pair.starting = false; }
}
function togglePair(pair) {
  pair.userPaused = pair.videos.some(video => !video.paused);
  pair.userStarted = true;
  updatePair(pair);
}
for (const pair of pairs) {
  pair.restart.addEventListener('click', () => {
    loadPair(pair);
    pair.videos.forEach(video => { video.currentTime = 0; });
    pair.userPaused = false;
    pair.userStarted = true;
    updatePair(pair);
  });
  pair.view.addEventListener('keydown', event => {
    if (event.target === pair.view && (event.code === 'Space' || event.code === 'Enter')) {
      event.preventDefault();
      togglePair(pair);
    }
  });
  pair.videos.forEach(video => {
    video.addEventListener('click', () => togglePair(pair));
    video.addEventListener('canplay', () => updatePair(pair));
    video.addEventListener('waiting', () => { if (wantsPlayback(pair)) pausePair(pair); });
    video.addEventListener('error', () => { pausePair(pair); pair.restart.title = 'Video unavailable'; });
  });
  pair.videos[0].addEventListener('timeupdate', () => {
    const [clock, follower] = pair.videos;
    if (follower.readyState >= 2 && Math.abs(clock.currentTime - follower.currentTime) > .08) follower.currentTime = clock.currentTime;
  });
}
const posterObserver = new IntersectionObserver(entries => {
  for (const entry of entries) {
    if (!entry.isIntersecting) continue;
    preparePosters(pairs.find(pair => pair.element === entry.target));
    posterObserver.unobserve(entry.target);
  }
}, { rootMargin: '350px 0px' });
const playObserver = new IntersectionObserver(entries => {
  for (const entry of entries) {
    const pair = pairs.find(pair => pair.view === entry.target);
    pair.inView = entry.isIntersecting;
    updatePair(pair);
  }
}, { threshold: .08 });
pairs.forEach(pair => { posterObserver.observe(pair.element); playObserver.observe(pair.view); });
document.addEventListener('visibilitychange', () => pairs.forEach(updatePair));
reducedMotion.addEventListener('change', () => {
  pairs.forEach(pair => { pair.userStarted = false; updatePair(pair); });
});
document.querySelector('#recorded-examples').addEventListener('toggle', () => pairs.forEach(updatePair));
document.querySelector('#copy-bibtex').addEventListener('click', async event => {
  const button = event.currentTarget;
  const status = document.querySelector('#copy-status');
  try {
    await navigator.clipboard.writeText(document.querySelector('#bibtex').textContent);
    button.textContent = 'Copied';
    status.textContent = 'BibTeX copied to clipboard.';
  } catch {
    const range = document.createRange();
    range.selectNodeContents(document.querySelector('#bibtex'));
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    status.textContent = 'BibTeX selected. Use your browser’s Copy command.';
    button.textContent = 'Selected';
  }
});
