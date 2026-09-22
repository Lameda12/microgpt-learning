/* ==========================================================================
   MicroGPT Portfolio — vanilla JS
   ========================================================================== */

(function () {
  'use strict';

  /* ---------- Navbar shadow on scroll ---------- */
  const nav = document.getElementById('mainNav');
  const onScroll = () => {
    if (window.scrollY > 12) nav.classList.add('scrolled');
    else nav.classList.remove('scrolled');
  };
  document.addEventListener('scroll', onScroll, { passive: true });
  onScroll();

  /* ---------- Reveal-on-scroll ---------- */
  const revealTargets = document.querySelectorAll(
    '.impl-card, .roadmap-step, .demo-window, .contribute-card, .code-window'
  );
  revealTargets.forEach((el) => el.classList.add('reveal'));

  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('is-visible');
          io.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.15 }
  );
  revealTargets.forEach((el) => io.observe(el));

  /* ---------- Animated stat counters ---------- */
  const counters = document.querySelectorAll('.stat-num');
  const animateCount = (el) => {
    const target = parseInt(el.dataset.count, 10);
    const duration = 1200;
    const start = performance.now();
    const step = (now) => {
      const progress = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      const value = Math.round(target * eased);
      el.textContent = value.toLocaleString();
      if (progress < 1) requestAnimationFrame(step);
      else el.textContent = target.toLocaleString();
    };
    requestAnimationFrame(step);
  };
  const counterIo = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          animateCount(entry.target);
          counterIo.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.6 }
  );
  counters.forEach((el) => counterIo.observe(el));

  /* ---------- Copy code snippet ---------- */
  const copyBtn = document.getElementById('copyCode');
  const snippet = document.getElementById('snippetCode');
  if (copyBtn && snippet) {
    copyBtn.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(snippet.innerText);
        const original = copyBtn.innerHTML;
        copyBtn.innerHTML = '<i class="bi bi-check2"></i> Copied';
        setTimeout(() => (copyBtn.innerHTML = original), 1600);
      } catch (err) {
        console.error('Clipboard copy failed', err);
      }
    });
  }

  /* ---------- In-browser character-level Markov demo ----------
     A tiny order-2 character Markov chain trained on a short
     public-domain Shakespeare excerpt, mirroring the char-level
     tokenization used by microgpt.py / shakespeare_gpt.py.
  ------------------------------------------------------------- */
  const CORPUS =
    "ROMEO: But soft, what light through yonder window breaks? " +
    "It is the east, and Juliet is the sun. " +
    "Arise, fair sun, and kill the envious moon, " +
    "Who is already sick and pale with grief. " +
    "JULIET: O Romeo, Romeo, wherefore art thou Romeo? " +
    "Deny thy father and refuse thy name; " +
    "Or if thou wilt not, be but sworn my love, " +
    "And I'll no longer be a Capulet. " +
    "HAMLET: To be, or not to be, that is the question: " +
    "Whether 'tis nobler in the mind to suffer " +
    "The slings and arrows of outrageous fortune, " +
    "Or to take arms against a sea of troubles.";

  function buildModel(corpus, order) {
    const model = {};
    for (let i = 0; i < corpus.length - order; i++) {
      const key = corpus.slice(i, i + order);
      const next = corpus[i + order];
      if (!model[key]) model[key] = [];
      model[key].push(next);
    }
    return model;
  }

  const ORDER = 3;
  const model = buildModel(CORPUS, ORDER);
  const keys = Object.keys(model);

  function generate(length) {
    let seed = keys[Math.floor(Math.random() * keys.length)];
    let out = seed;
    for (let i = 0; i < length; i++) {
      const key = out.slice(-ORDER);
      const options = model[key];
      if (!options || !options.length) break;
      out += options[Math.floor(Math.random() * options.length)];
    }
    return out;
  }

  const demoOutput = document.getElementById('demoOutput');
  const demoRun = document.getElementById('demoRun');
  const demoReset = document.getElementById('demoReset');
  const demoStatus = document.getElementById('demoStatus');
  let typeTimer = null;

  function typeOut(text) {
    clearInterval(typeTimer);
    demoOutput.innerHTML = '';
    let i = 0;
    demoStatus.textContent = 'sampling…';
    demoRun.disabled = true;
    typeTimer = setInterval(() => {
      demoOutput.textContent = text.slice(0, i) ;
      demoOutput.innerHTML += '<span class="cursor">&#9608;</span>';
      i++;
      if (i > text.length) {
        clearInterval(typeTimer);
        demoStatus.textContent = 'done — ' + text.length + ' tokens';
        demoRun.disabled = false;
      }
    }, 28);
  }

  if (demoRun) {
    demoRun.addEventListener('click', () => {
      const text = generate(180);
      typeOut(text);
    });
  }
  if (demoReset) {
    demoReset.addEventListener('click', () => {
      clearInterval(typeTimer);
      demoOutput.innerHTML = '<span class="cursor">&#9608;</span>';
      demoStatus.textContent = 'idle';
      demoRun.disabled = false;
    });
  }
})();
