/* Start each comparison grid together; synchronize the two views in each pair. */
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const groups = [...document.querySelectorAll('.demo-group')].map(element => ({
  element,
  grid: element.querySelector('.comparison-grid'),
  pairs: [...element.querySelectorAll('.demo')].map(pair => ({
    view: pair.querySelector('.video-pair'), videos: [...pair.querySelectorAll('video')],
  })),
  videos: [...element.querySelectorAll('video')],
  restart: element.querySelector('.group-replay'),
  inView: false, userPaused: false, userStarted: false, loaded: false, starting: false,
}));

function wantsPlayback(group) {
  const disclosure = group.element.closest('details');
  return group.inView && !document.hidden && (!disclosure || disclosure.open)
    && !group.userPaused && (!reducedMotion.matches || group.userStarted);
}
function preparePosters(group) {
  group.videos.forEach(video => { if (!video.poster) video.poster = video.dataset.poster; });
}
function loadGroup(group) {
  if (group.loaded) return;
  group.loaded = true;
  preparePosters(group);
  group.videos.forEach(video => {
    video.muted = true;
    video.src = video.dataset.src;
    video.load();
  });
}
function pauseGroup(group) { group.videos.forEach(video => video.pause()); }
async function updateGroup(group) {
  if (!wantsPlayback(group)) { pauseGroup(group); return; }
  loadGroup(group);
  // All eight views must be ready before the grid starts.
  if (group.starting || group.videos.some(video => video.readyState < 3)) return;
  if (group.videos.every(video => !video.paused)) return;
  group.starting = true;
  group.pairs.forEach(({videos:[clock,follower]}) => {
    if (Math.abs(clock.currentTime-follower.currentTime)>.08) follower.currentTime=clock.currentTime;
  });
  try {
    await Promise.all(group.videos.map(video => video.play()));
    if (!wantsPlayback(group)) pauseGroup(group);
  } catch {
    // A user gesture on Replay or either view also starts the complete grid.
    pauseGroup(group);
  } finally { group.starting = false; }
}
function toggleGroup(group) {
  group.userPaused = group.videos.some(video => !video.paused);
  group.userStarted = true;
  updateGroup(group);
}
for (const group of groups) {
  group.restart.addEventListener('click', () => {
    loadGroup(group);
    group.videos.forEach(video => { video.currentTime=0; });
    group.userPaused=false;
    group.userStarted=true;
    updateGroup(group);
  });
  group.pairs.forEach(pair => {
    pair.view.addEventListener('keydown', event => {
      if (event.target===pair.view && (event.code==='Space'||event.code==='Enter')) {
        event.preventDefault(); toggleGroup(group);
      }
    });
    pair.videos[0].addEventListener('timeupdate', () => {
      const [clock,follower]=pair.videos;
      if(follower.readyState>=2 && Math.abs(clock.currentTime-follower.currentTime)>.08) follower.currentTime=clock.currentTime;
    });
  });
  group.videos.forEach(video => {
    video.addEventListener('click',()=>toggleGroup(group));
    video.addEventListener('canplay',()=>updateGroup(group));
    video.addEventListener('waiting',()=>{if(wantsPlayback(group))pauseGroup(group);});
    video.addEventListener('error',()=>{pauseGroup(group);group.restart.title='Video unavailable';});
  });
}
const posterObserver=new IntersectionObserver(entries=>{
  entries.forEach(entry=>{
    if(!entry.isIntersecting)return;
    preparePosters(groups.find(group=>group.element===entry.target));
    posterObserver.unobserve(entry.target);
  });
},{rootMargin:'350px 0px'});
const playObserver=new IntersectionObserver(entries=>{
  entries.forEach(entry=>{
    const group=groups.find(group=>group.grid===entry.target);
    group.inView=entry.isIntersecting;
    updateGroup(group);
  });
},{threshold:.02});
groups.forEach(group=>{posterObserver.observe(group.element);playObserver.observe(group.grid);});
document.addEventListener('visibilitychange',()=>groups.forEach(updateGroup));
reducedMotion.addEventListener('change',()=>{
  groups.forEach(group=>{group.userStarted=false;updateGroup(group);});
});
document.querySelector('#more-examples').addEventListener('toggle',()=>groups.forEach(updateGroup));
document.querySelector('#copy-bibtex').addEventListener('click',async event=>{
  const button=event.currentTarget;
  const status=document.querySelector('#copy-status');
  try {
    await navigator.clipboard.writeText(document.querySelector('#bibtex').textContent);
    button.textContent='Copied'; status.textContent='BibTeX copied to clipboard.';
  } catch {
    const range=document.createRange();range.selectNodeContents(document.querySelector('#bibtex'));
    const selection=window.getSelection();selection.removeAllRanges();selection.addRange(range);
    status.textContent='BibTeX selected. Use your browser’s Copy command.';button.textContent='Selected';
  }
});
