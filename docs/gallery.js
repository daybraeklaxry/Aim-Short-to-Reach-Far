const examples = JSON.parse(document.querySelector('#demo-data').textContent);
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');

class ComparisonPlayer {
  constructor(root) {
    this.root = root;
    this.method = root.dataset.method;
    this.task = 'pusht';
    this.example = 0;
    this.videos = [...root.querySelectorAll('video')];
    this.button = root.querySelector('.play-button');
    this.timeline = root.querySelector('.timeline');
    this.counter = root.querySelector('.action-count');
    this.message = root.querySelector('.player-message');
    this.inView = false;
    this.userPaused = false;
    this.userStarted = false;
    this.loaded = false;
    this.starting = false;
    this.generation = 0;
    this.setCase();
    root.querySelectorAll('[data-task]').forEach(button => {
      button.addEventListener('click', () => this.changeTask(button.dataset.task));
      button.addEventListener('keydown', event => {
        const tabs = [...root.querySelectorAll('[data-task]')];
        const index = tabs.indexOf(button);
        let next;
        if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
        if (event.key === 'ArrowLeft') next = (index - 1 + tabs.length) % tabs.length;
        if (event.key === 'Home') next = 0;
        if (event.key === 'End') next = tabs.length - 1;
        if (next === undefined) return;
        event.preventDefault();
        tabs[next].focus();
        this.changeTask(tabs[next].dataset.task);
      });
    });
    root.querySelectorAll('[data-example]').forEach(button => button.addEventListener('click', () => {
      this.example = Number(button.dataset.example);
      this.setCase();
      this.load();
    }));
    this.button.addEventListener('click', () => {
      this.userStarted = true;
      this.userPaused = this.videos.some(video => !video.paused);
      this.load();
      this.updatePlayback();
    });
    root.querySelector('.replay-button').addEventListener('click', () => {
      this.userStarted = true;
      this.userPaused = false;
      this.load();
      this.seek(0);
      this.updatePlayback();
    });
    this.timeline.addEventListener('input', () => this.seek(Number(this.timeline.value) / 1000));
    this.videos.forEach(video => {
      video.addEventListener('canplay', () => { this.updateSeekAvailability(); this.updatePlayback(); });
      video.addEventListener('progress', () => this.updateSeekAvailability());
      video.addEventListener('canplaythrough', () => this.updateSeekAvailability());
      video.addEventListener('play', () => this.setButton(true));
      video.addEventListener('pause', () => this.setButton(false));
      video.addEventListener('error', () => {
        this.pause();
        this.message.textContent = 'The video could not load. Choose another example or reload the page.';
      });
    });
    this.videos[0].addEventListener('timeupdate', () => this.updateProgress());
    this.videos[0].addEventListener('ended', () => {
      if (!this.wantsPlayback()) { this.pause(); return; }
      this.pause();
      this.seek(0);
      this.updatePlayback();
    });
  }
  changeTask(task) {
    if (this.task === task) return;
    this.task = task;
    this.example = 0;
    this.setCase();
    this.load();
  }
  setCase() {
    this.pause();
    this.generation++;
    this.starting = false;
    this.loaded = false;
    this.case = examples[this.method][this.task][this.example];
    this.message.textContent = '';
    this.root.querySelectorAll('[data-task]').forEach(button => {
      const selected = button.dataset.task === this.task;
      button.setAttribute('aria-selected', String(selected));
      button.tabIndex = selected ? 0 : -1;
    });
    this.root.querySelector('.demo-panel').setAttribute('aria-labelledby', `${this.method}-tab-${this.task}`);
    this.root.querySelectorAll('[data-example]').forEach(button => button.setAttribute('aria-pressed', String(Number(button.dataset.example) === this.example)));
    this.videos.forEach((video, i) => {
      const side = i ? 'ap' : 'baseline';
      video.poster = `assets/comparisons/${this.case.name}-${side}.jpg`;
      video.dataset.source = `assets/comparisons/${this.case.name}-${side}.mp4`;
      const method = i ? (this.method === 'cem' ? 'AP-CEM' : 'AP-rank') : (this.method === 'cem' ? 'Final-goal CEM' : 'Direct');
      video.setAttribute('aria-label', `${method} ${this.case.label} example ${this.example + 1}`);
      const success = this.case[`${side}Success`];
      const steps = this.case[`${side}Steps`];
      const result = this.root.querySelector(`[data-result=${side}]`);
      result.classList.toggle('success', success);
      result.classList.toggle('failure', !success);
      result.querySelector('.status-text').textContent = success ? 'Success' : 'Failure';
      result.querySelector('.finish-step').textContent = `${steps} actions`;
      result.setAttribute('aria-label', `${method}: ${success ? 'success' : 'failure'} after ${steps} actions`);
      result.querySelector('svg').innerHTML = success ? '<path d="m5 12 4 4L19 6"/>' : '<path d="m6 6 12 12M18 6 6 18"/>';
    });
    const goal = this.root.querySelector('.goal-image');
    goal.src = `assets/demos/${this.case.name}-goal.jpg`;
    goal.alt = `Shared ${this.case.label} goal for example ${this.example + 1}`;
    this.root.querySelector('.goal-link').href = goal.src;
    this.root.querySelector('.goal-description').textContent = this.case.goal;
    this.root.querySelector('.case-summary').textContent = this.case.baselineSuccess
      ? 'Both methods reach the goal in this example.'
      : `${this.method === 'cem' ? 'AP-CEM' : 'AP-rank'} reaches the goal. ${this.method === 'cem' ? 'Final-goal CEM' : 'Direct'} does not.`;
    this.timeline.value = 0;
    this.timeline.disabled = true;
    this.updateProgress(true);
  }
  load() {
    if (this.loaded) return;
    this.loaded = true;
    this.videos.forEach(video => {
      video.muted = true;
      video.preload = 'auto';
      video.src = video.dataset.source;
      video.load();
    });
  }
  wantsPlayback() {
    return this.inView && !document.hidden && !this.userPaused && (!reducedMotion.matches || this.userStarted);
  }
  async updatePlayback() {
    if (!this.wantsPlayback()) { this.pause(); return; }
    this.load();
    if (this.starting || this.videos.some(video => video.readyState < 3) || this.videos.every(video => !video.paused)) return;
    const generation = this.generation;
    this.starting = true;
    const [clock, follower] = this.videos;
    if (Math.abs(clock.currentTime - follower.currentTime) > .1) follower.currentTime = clock.currentTime;
    try {
      await Promise.all(this.videos.map(video => video.play()));
      if (generation === this.generation && !this.wantsPlayback()) this.pause();
    } catch {
      if (generation === this.generation) this.pause();
    } finally {
      if (generation === this.generation) this.starting = false;
    }
  }
  pause() { this.videos.forEach(video => video.pause()); this.setButton(false); }
  setButton(playing) {
    this.button.querySelector('span').textContent = playing ? 'Pause' : 'Play';
    this.button.setAttribute('aria-label', `${playing ? 'Pause' : 'Play'} ${this.method === 'cem' ? 'AP-CEM' : 'AP-rank'} comparison`);
    this.button.querySelector('svg').innerHTML = playing ? '<path d="M8 5v14M16 5v14"/>' : '<path d="m9 5 11 7-11 7z"/>';
  }
  updateSeekAvailability() {
    this.timeline.disabled = !this.videos.every(video => video.seekable.length && video.seekable.end(0) > 0);
  }
  seek(fraction) {
    this.videos.forEach(video => { if (Number.isFinite(video.duration)) video.currentTime = Math.min(fraction * video.duration, Math.max(0, video.duration - .05)); });
    this.updateProgress();
  }
  updateProgress(reset = false) {
    const [clock, follower] = this.videos;
    const time = reset ? 0 : clock.currentTime || 0;
    const total = Math.max(this.case.baselineSteps, this.case.apSteps);
    const action = Math.min(total, Math.floor(time * this.case.actionsPerSecond));
    if (!reset && Number.isFinite(clock.duration)) this.timeline.value = Math.round(time / clock.duration * 1000);
    this.counter.textContent = `Action ${action}`;
    this.timeline.setAttribute('aria-valuetext', `Action ${action} of ${total}`);
    for (const side of ['baseline', 'ap']) {
      const outcome = this.root.querySelector(`[data-outcome=${side}]`);
      const steps = this.case[`${side}Steps`];
      const ended = action >= steps;
      outcome.textContent = ended ? (this.case[`${side}Success`] ? 'Goal reached' : 'Goal not reached') : `Action ${Math.min(action, steps)} of ${steps}`;
      outcome.classList.toggle('reached', ended && this.case[`${side}Success`]);
    }
    if (!reset && follower.readyState >= 2 && Math.abs(time - follower.currentTime) > .12) follower.currentTime = time;
  }
}

const players = [...document.querySelectorAll('.demo-player')].map(root => new ComparisonPlayer(root));
const visibility = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    const player = players.find(player => player.root === entry.target);
    player.inView = entry.isIntersecting;
    if (entry.isIntersecting) player.load();
    player.updatePlayback();
  });
}, {threshold:.12});
players.forEach(player => visibility.observe(player.root));
document.addEventListener('visibilitychange', () => players.forEach(player => player.updatePlayback()));
reducedMotion.addEventListener('change', () => players.forEach(player => { player.userStarted = false; player.updatePlayback(); }));

document.querySelector('#copy-bibtex').addEventListener('click', async event => {
  const button = event.currentTarget;
  const code = document.querySelector('#bibtex');
  const status = document.querySelector('#copy-status');
  try {
    await navigator.clipboard.writeText(code.textContent);
    button.querySelector('span').textContent = 'Copied';
    status.textContent = 'BibTeX copied to clipboard.';
  } catch {
    const range = document.createRange();
    range.selectNodeContents(code);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    button.querySelector('span').textContent = 'Text selected';
    status.textContent = 'Use your browser’s Copy command to copy the selected BibTeX.';
  }
});
