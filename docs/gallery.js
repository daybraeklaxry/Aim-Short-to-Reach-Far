/* Original camera crops, played on a shared clock. */
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const pairs = [...document.querySelectorAll('.demo')].map(element => ({
  element, videos: [...element.querySelectorAll('video')],
  toggle: element.querySelector('.pair-toggle'), seek: element.querySelector('.pair-seek'),
  inView: false, userPaused: false, userStarted: false, loaded: false, starting: false,
}));

function wantsPlayback(pair) {
  const disclosure = pair.element.closest('.more-examples');
  return pair.inView && !document.hidden && (!disclosure || disclosure.open)
    && !pair.userPaused && (!reducedMotion.matches || pair.userStarted);
}
function preparePosters(pair) {
  for (const video of pair.videos) if (!video.poster) video.poster = video.dataset.poster;
}
function loadPair(pair) {
  if (pair.loaded) return;
  pair.loaded = true;
  preparePosters(pair);
  for (const video of pair.videos) {
    video.muted = true;
    video.src = video.dataset.src;
    video.load();
  }
}
function pausePair(pair) {
  pair.videos.forEach(video => video.pause());
  pair.toggle.textContent = 'Play comparison';
  pair.toggle.setAttribute('aria-pressed', 'false');
}
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
    if (wantsPlayback(pair)) {
      pair.toggle.textContent = 'Pause comparison';
      pair.toggle.setAttribute('aria-pressed', 'true');
    } else pausePair(pair);
  } catch {
    // A browser can block autoplay; the control starts both on a user gesture.
    pausePair(pair);
  } finally { pair.starting = false; }
}
for (const pair of pairs) {
  pair.toggle.setAttribute('aria-pressed', 'false');
  pair.toggle.addEventListener('click', () => {
    pair.userPaused = pair.videos.some(video => !video.paused);
    pair.userStarted = true;
    pair.inView = true;
    updatePair(pair);
  });
  pair.element.querySelector('.pair-restart').addEventListener('click', () => {
    loadPair(pair);
    pair.videos.forEach(video => { video.currentTime = 0; });
    pair.seek.value = 0;
    pair.userPaused = false;
    pair.userStarted = true;
    updatePair(pair);
  });
  pair.seek.addEventListener('input', () => {
    loadPair(pair);
    const duration = pair.videos[0].duration;
    if (Number.isFinite(duration)) pair.videos.forEach(video => { video.currentTime = duration * pair.seek.value / 1000; });
  });
  pair.videos.forEach(video => {
    video.addEventListener('canplay', () => updatePair(pair));
    // Hold both views if either crop buffers.
    video.addEventListener('waiting', () => { if (wantsPlayback(pair)) pausePair(pair); });
    video.addEventListener('error', () => {
      pausePair(pair);
      pair.toggle.textContent = 'Video unavailable';
      pair.toggle.disabled = true;
    });
  });
  pair.videos[0].addEventListener('timeupdate', () => {
    const [clock, follower] = pair.videos;
    if (Number.isFinite(clock.duration)) pair.seek.value = clock.currentTime / clock.duration * 1000;
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
    const pair = pairs.find(pair => pair.element.querySelector('.video-pair') === entry.target);
    pair.inView = entry.isIntersecting;
    updatePair(pair);
  }
}, { threshold: .08 });
pairs.forEach(pair => {
  posterObserver.observe(pair.element);
  playObserver.observe(pair.element.querySelector('.video-pair'));
});
document.addEventListener('visibilitychange', () => pairs.forEach(updatePair));
reducedMotion.addEventListener('change', () => {
  pairs.forEach(pair => { pair.userStarted = false; updatePair(pair); });
});
document.querySelector('#more-examples').addEventListener('toggle', () => pairs.forEach(updatePair));
document.querySelector('#copy-bibtex').addEventListener('click', async event => {
  const button = event.currentTarget;
  const status = document.querySelector('#copy-status');
  try {
    await navigator.clipboard.writeText(document.querySelector('#bibtex').textContent);
    button.textContent = 'Copied';
    status.textContent = 'BibTeX copied to clipboard.';
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(document.querySelector('#bibtex'));
    selection.removeAllRanges();
    selection.addRange(range);
    status.textContent = 'BibTeX selected. Use your browser’s Copy command.';
    button.textContent = 'Selected';
  }
});
