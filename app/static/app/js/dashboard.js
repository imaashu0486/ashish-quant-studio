/* dashboard.js — full replacement (tooltip + index + predict form + theme sync) */
(function(){
  // Ensure DOM ready
  document.addEventListener('DOMContentLoaded', function(){

    // Enforce default theme (if nothing saved yet) to dark for best look
    try {
      const root = document.documentElement;
      const saved = localStorage.getItem('aqs-theme');
      if (!saved) {
        // set default to dark but don't override existing explicit user setting
        root.setAttribute('data-theme', root.getAttribute('data-theme') || 'dark');
      }
    } catch(e){}

    // keyboard quick toggle: Shift + T (only when NOT typing in an input/textarea)
    (function(){
      const root = document.documentElement;
      document.addEventListener('keydown', (e) => {
        // ignore if typing in an input, textarea or contenteditable
        const active = document.activeElement;
        const typing = active && (
          active.tagName === 'INPUT' ||
          active.tagName === 'TEXTAREA' ||
          active.isContentEditable
        );
        if (typing) return;

        // require SHIFT + T
        if (e.key === 'T' && e.shiftKey && !e.metaKey && !e.ctrlKey && !e.altKey) {
          e.preventDefault();
          const cur = root.getAttribute('data-theme') || 'dark';
          const next = cur === 'dark' ? 'light' : 'dark';
          root.setAttribute('data-theme', next);
          try { localStorage.setItem('aqs-theme', next); } catch (err) {}
          // if you have a function to restyle charts or other UI, call it here:
          if (typeof restylePlotlyCharts === 'function') try { restylePlotlyCharts(); } catch(e) {}
        }
      });
    })();


    // UTILS
    function toText(el) { return el ? (el.innerText || el.textContent || '').trim() : ''; }

    /* ===================== Tooltip helper ===================== */
    /* ------------------ REPLACE tooltip implementation with this hardened version ------------------ */
/* Put this near the top of your dashboard.js (or replace the existing tooltip functions) */

(function(){
  // robust tooltip used for index cards — always attached to document.body
  let aqsTooltip = null;
  function createAqsTooltip() {
    if (aqsTooltip && document.body.contains(aqsTooltip)) return aqsTooltip;
    aqsTooltip = document.createElement('div');
    aqsTooltip.className = 'aqs-tooltip';
    // Important: append to body so it's never clipped by local overflow
    document.body.appendChild(aqsTooltip);

    // force fixed positioning and super high stacking order
    const s = aqsTooltip.style;
    s.position = 'fixed';
    s.zIndex = '2147483647'; // max 32-bit signed int
    s.pointerEvents = 'none';
    s.left = '0px';
    s.top = '0px';
    s.opacity = '0';
    s.transform = 'translateY(6px) scale(.98)';
    s.transition = 'opacity .12s ease, transform .12s ease';
    return aqsTooltip;
  }

  function showAqsTooltip(html, clientX, clientY) {
    const tip = createAqsTooltip();
    tip.innerHTML = html;

    // make sure it is rendered (so getBoundingClientRect will be correct)
    tip.style.opacity = '0';
    tip.classList.remove('show');

    // small microtask delay so browser lays out tooltip contents
    requestAnimationFrame(() => {
      const rectTip = tip.getBoundingClientRect();

      // position horizontally centered on clientX, and above clientY by default
      let left = Math.round(clientX - rectTip.width / 2);
      let top = Math.round(clientY - rectTip.height - 12);

      // clamp inside viewport with 8px padding
      const pad = 8;
      if (left < pad) left = pad;
      if (left + rectTip.width > window.innerWidth - pad) left = Math.max(pad, window.innerWidth - rectTip.width - pad);
      if (top < pad) {
        // if not enough space above, display below the anchor
        top = Math.round(clientY + 12);
      }

      // apply position
      tip.style.left = left + 'px';
      tip.style.top = top + 'px';

      // show
      tip.classList.add('show');
      tip.style.opacity = '1';
      tip.style.transform = 'translateY(0) scale(1)';
    });
  }

  function hideAqsTooltip() {
    if (!aqsTooltip) return;
    aqsTooltip.classList.remove('show');
    aqsTooltip.style.opacity = '0';
    aqsTooltip.style.transform = 'translateY(6px) scale(.98)';
    // remove element after animation so next create will be clean
    setTimeout(()=> {
      if (aqsTooltip && aqsTooltip.parentNode) {
        aqsTooltip.parentNode.removeChild(aqsTooltip);
      }
      aqsTooltip = null;
    }, 180);
  }

  // expose functions the rest of your file can call (attach to window for compatibility)
  window.showAqsTooltip = showAqsTooltip;
  window.hideAqsTooltip = hideAqsTooltip;
})();

    /* ===================== Indices track & cards ===================== */
    const track = document.getElementById('topIndicesTrack');

// ===== Strict index binding: attach tooltip ONLY to true Top Indices cards =====
// ===== Strict + defensive index binding: only show tooltip when the element *looks* like a Top Index card =====
if (track) {
  // select direct children but be defensive if heatmap inserts similar nodes
  const cards = Array.from(track.querySelectorAll(':scope > .idx-card'));

  cards.forEach(card => {
    // keep click navigation for all cards (so heatmap tiles still clickable)
    card.addEventListener('click', () => {
      const sym = card.dataset.symbol || '';
      if (sym) window.location.href = `/predict/${encodeURIComponent(sym)}/1/`;
    });

    // keyboard support for click
    card.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); card.click(); }
    });

    // tooltip timers
    let enterTimer = null;
    let touchTimer = null;

    // Helper: check if this node actually looks like a TOP-INDEX card we want a tooltip for
    function isRealIndexCard(el) {
      // require both name and last-price nodes to be present
      const hasName = !!el.querySelector('.idx-name');
      const hasLast = !!el.querySelector('.idx-last');
      // require change or pct optionally (not mandatory), but if neither present it's not an index card
      const hasChange = !!el.querySelector('.idx-change') || !!el.querySelector('.idx-pct');
      return !!(hasName && hasLast && hasChange);
    }

    // MOUSE: show tooltip after short delay, but only when element *is* an index card
    card.addEventListener('mouseenter', (ev) => {
      if (enterTimer) clearTimeout(enterTimer);
      enterTimer = setTimeout(() => {
        // Defensive: abort if this isn't a real index card
        if (!isRealIndexCard(card)) return;
        const nameEl = card.querySelector('.idx-name');
        const lastEl = card.querySelector('.idx-last');
        const changeEl = card.querySelector('.idx-change');
        const pctEl = card.querySelector('.idx-pct');

        const name = nameEl ? (nameEl.innerText || nameEl.textContent || '') : '';
        const last = lastEl ? (lastEl.innerText || lastEl.textContent || '') : '';
        const change = changeEl ? (changeEl.innerText || changeEl.textContent || '') : '';
        const pct = pctEl ? (pctEl.innerText || pctEl.textContent || '') : '';

        const html = `<div class="t-title">${name}</div><div class="t-row"><strong>${last}</strong> · ${change} · ${pct}</div>`;
        const b = card.getBoundingClientRect();
        showAqsTooltip(html, b.left + (b.width / 2), b.top);
      }, 120);
    });

    card.addEventListener('mouseleave', (ev) => {
      if (enterTimer) { clearTimeout(enterTimer); enterTimer = null; }
      hideAqsTooltip();
    });

    // TOUCH: long-press shows tooltip only for real index cards
    card.addEventListener('touchstart', (e) => {
      if (touchTimer) { clearTimeout(touchTimer); touchTimer = null; }
      touchTimer = setTimeout(() => {
        if (!isRealIndexCard(card)) return;
        const nameEl = card.querySelector('.idx-name');
        const lastEl = card.querySelector('.idx-last');
        const changeEl = card.querySelector('.idx-change');
        const pctEl = card.querySelector('.idx-pct');

        const name = nameEl ? (nameEl.innerText || nameEl.textContent || '') : '';
        const last = lastEl ? (lastEl.innerText || lastEl.textContent || '') : '';
        const change = changeEl ? (changeEl.innerText || changeEl.textContent || '') : '';
        const pct = pctEl ? (pctEl.innerText || pctEl.textContent || '') : '';

        const html = `<div class="t-title">${name}</div><div class="t-row"><strong>${last}</strong> · ${change} · ${pct}</div>`;
        const b = card.getBoundingClientRect();
        showAqsTooltip(html, b.left + (b.width / 2), b.top);
      }, 380);
    }, { passive: true });

    card.addEventListener('touchend', (e) => {
      if (touchTimer) { clearTimeout(touchTimer); touchTimer = null; }
      hideAqsTooltip();
    });

    // ensure tooltip hidden if page scrolled or resized while visible
    window.addEventListener('scroll', () => { if (enterTimer) { clearTimeout(enterTimer); enterTimer = null; } hideAqsTooltip(); }, { passive: true });
    window.addEventListener('resize', () => { if (enterTimer) { clearTimeout(enterTimer); enterTimer = null; } hideAqsTooltip(); });
  });
}

    // scroll buttons (if present)
    const leftBtn = document.getElementById('idx-scroll-left');
    const rightBtn = document.getElementById('idx-scroll-right');
    if (track && leftBtn && rightBtn){
      const amt = Math.round(window.innerWidth * 0.6);
      leftBtn.addEventListener('click', ()=> track.scrollBy({left:-amt, behavior:'smooth'}));
      rightBtn.addEventListener('click', ()=> track.scrollBy({left:amt, behavior:'smooth'}));
    }

    /* ===================== Movers rows clickable ===================== */
    document.querySelectorAll('.mover-row').forEach(row=>{
      row.addEventListener('click', (e)=>{
        if (e.target.tagName.toLowerCase() === 'a') return;
        const sym = row.dataset.symbol || '';
        if (sym) window.location.href = `/predict/${encodeURIComponent(sym)}/1/`;
      });
    });

    /* ===================== Heatmap tiles binding ===================== */
    function bindTiles(root){
      if (!root) return;
      const tiles = root.querySelectorAll('.tile');
      tiles.forEach(t=>{
        t.addEventListener('click', ()=> {
          const sym = t.dataset.symbol || t.getAttribute('data-symbol') || '';
          if (sym) window.location.href = `/predict/${encodeURIComponent(sym)}/1/`;
        });
      });
    }
    bindTiles(document.getElementById('hm'));
    bindTiles(document.getElementById('leaders'));

    /* ===================== Predict form normalization ===================== */
    const predictForm = document.getElementById('predictForm');
    if (predictForm){
      const input = predictForm.querySelector('#ticker');
      const tpl = predictForm.getAttribute('data-action-template') || predictForm.getAttribute('action') || '';
      predictForm.addEventListener('submit', function(e){
        const t = (input.value || '').trim().toUpperCase();
        if (!t){ e.preventDefault(); input.focus(); return; }
        if (tpl && tpl.includes('TICK')){
          this.setAttribute('action', tpl.replace('TICK', encodeURIComponent(t)));
        } else {
          // fallback append to existing
          // keep original behavior otherwise
        }
      });
    }

    /* ===================== Skeletons for empty charts ===================== */
    document.querySelectorAll('.leaders, .plotly-container').forEach(cont=>{
      if (cont && cont.innerHTML.trim() === ''){
        const skeleton = document.createElement('div');
        skeleton.className = 'panel';
        skeleton.innerHTML = '<div style="height:120px;display:flex;align-items:center;justify-content:center;color:rgba(255,255,255,0.12)">Loading chart…</div>';
        cont.appendChild(skeleton);
      } else {
        // ensure plot containers background is transparent for theme
        const inner = cont.querySelector('div');
        if (inner) inner.style.background = 'transparent';
      }
    });

    /* ===================== Result page buttons (copy/toggle) ===================== */
    const copyBtn = document.getElementById('copyLink');
    if (copyBtn){
      copyBtn.addEventListener('click', async ()=> {
        try {
          await navigator.clipboard.writeText(window.location.href);
          copyBtn.textContent = 'Link copied ✓';
          setTimeout(()=> copyBtn.textContent = 'Copy result link', 1800);
        } catch (e) { window.prompt('Copy this link', window.location.href); }
      });
    }
    const toggleBtn = document.getElementById('toggleFeatures');
    const featuresPanel = document.getElementById('featuresPanel');
    if (toggleBtn && featuresPanel){
      toggleBtn.addEventListener('click', ()=> {
        if (featuresPanel.style.display === 'none') { featuresPanel.style.display = ''; toggleBtn.textContent = 'Hide feature table'; }
        else { featuresPanel.style.display = 'none'; toggleBtn.textContent = 'Show feature table'; }
      });
    }

    /* ===================== Mutation observer to bind new tiles/idx-cards dynamically ===================== */
    const observer = new MutationObserver((mutList)=> {
      mutList.forEach(m => {
        m.addedNodes.forEach(n => {
          if (n.nodeType !== 1) return;
          bindTiles(n);
          if (n.matches && n.matches('.idx-card')) {
            // if entire idx-card added, re-run color logic quickly
            const right = n.querySelector('.idx-right'); const changeEl = n.querySelector('.idx-change');
            if (right && changeEl) {
              const txt = toText(changeEl);
              if (!right.classList.contains('green') && !right.classList.contains('red')) {
                if (txt.includes('▲') || txt.startsWith('+')) right.classList.add('green');
                else if (txt.includes('▼') || txt.includes('-')) right.classList.add('red');
              }
            }
          } else {
            // also check for idx-cards inside new node
            const innerCards = n.querySelectorAll && n.querySelectorAll('.idx-card') || [];
            innerCards.forEach(c => {
              const right = c.querySelector('.idx-right'); const changeEl = c.querySelector('.idx-change');
              if (right && changeEl) {
                const txt = toText(changeEl);
                if (!right.classList.contains('green') && !right.classList.contains('red')) {
                  if (txt.includes('▲') || txt.startsWith('+')) right.classList.add('green');
                  else if (txt.includes('▼') || txt.includes('-')) right.classList.add('red');
                }
              }
            });
          }
        });
      });
    });
    observer.observe(document.body, { childList: true, subtree: true });

  }); // DOMContentLoaded end
})();

/* ---------- Post-load fixer: color mover rows & defensive tooltip ---------- */
(function(){
  function applyRowColorsOnce(root=document){
    const rows = root.querySelectorAll && root.querySelectorAll('.movelist .mover-row') || [];
    rows.forEach(row=>{
      try{
        // get percent text: look for .mover-meta or last text node
        const metaEl = row.querySelector('.mover-meta');
        let txt = metaEl ? (metaEl.innerText || metaEl.textContent || '') : row.innerText || '';
        txt = txt.trim();
        // normalize: e.g. "2,484.50 · 4.97%" or "-0.12%" or " -0.12%"
        // look for minus sign or percentage numeric sign
        if (/[-−]/.test(txt)) {
          row.classList.remove('green'); row.classList.add('red');
        } else if (/[\+\u25B2]|\d+(\.\d+)?\s*%/.test(txt)) {
          // plus sign, upward arrow, or positive percent
          // attempt to detect explicit '+' or not negative
          // if there's a percent number, check numeric value
          const match = txt.match(/(-?\d+(\.\d+)?)\s*%/);
          if (match){
            const val = parseFloat(match[1].replace(/,/g,''));
            if (!isNaN(val)){
              if (val < 0){ row.classList.remove('green'); row.classList.add('red'); }
              else if (val > 0){ row.classList.remove('red'); row.classList.add('green'); }
            }
          } else {
            // fallback to adding green when no minus sign
            if (!row.classList.contains('red')) row.classList.add('green');
          }
        }
      }catch(e){}
    });
  }

  // run once on load
  document.addEventListener('DOMContentLoaded', ()=> applyRowColorsOnce(document));

  // watch for dynamically-added nodes (e.g. if your server injects lists later)
  const observer = new MutationObserver((mutList)=>{
    mutList.forEach(m=>{
      m.addedNodes.forEach(n=>{
        if (n.nodeType !== 1) return;
        // if entire mover list added, recolor
        if (n.matches && n.matches('.movelist')) applyRowColorsOnce(n);
        else if (n.querySelector && n.querySelector('.movelist')) applyRowColorsOnce(n);
        // if individual mover-row added
        if (n.matches && n.matches('.mover-row')) applyRowColorsOnce(n.parentNode || document);
      });
    });
  });
  observer.observe(document.body, {childList:true, subtree:true});

  // Defensive: ensure tooltip created uses fixed positioning (if some other code re-creates it)
  const oldCreate = window.createTooltip;
  window.createTooltip = function(){
    // if you had a global createTooltip, preserve it; otherwise ensure fixed tooltip
    try {
      if (oldCreate) oldCreate();
    } catch(e){}
    // If tooltip exists in DOM, enforce fixed & z-index
    const t = document.querySelector('.aqs-tooltip');
    if (t){
      t.style.position = 'fixed';
      t.style.zIndex = '2147483647';
      t.style.pointerEvents = 'none';
    }
  };
})();

/* ---------- Remote ticker suggester (uses /autocomplete/?q=) ---------- */
(function(){
  function debounce(fn, ms){ let t; return (...a)=>{ clearTimeout(t); t=setTimeout(()=>fn(...a), ms); }; }

  document.addEventListener('DOMContentLoaded', function(){
    const input = document.querySelector('#ticker');
    if (!input) return;

    // build dropdown container
    const sugg = document.createElement('div');
    sugg.className = 'ticker-suggester';
    const ul = document.createElement('ul');
    sugg.appendChild(ul);
    document.body.appendChild(sugg);

    let visible = false, active = -1, items = [];

    function position(){
      const r = input.getBoundingClientRect();
      sugg.style.left = (r.left + window.scrollX) + 'px';
      sugg.style.top  = (r.bottom + window.scrollY + 6) + 'px';
      sugg.style.minWidth = r.width + 'px';
    }
    function hide(){ sugg.style.display='none'; visible=false; active=-1; }
    function show(){ sugg.style.display='block'; visible=true; position(); }

    function render(list){
      ul.innerHTML='';
      list.forEach((item,i)=>{
        const li=document.createElement('li');
        li.innerHTML=`<span>${item.label}</span><span class="meta">${item.meta||''}</span>`;
        li.addEventListener('mouseenter',()=>setActive(i));
        li.addEventListener('click',()=>accept(i));
        ul.appendChild(li);
      });
      items=list; if(list.length) show(); else hide();
    }
    function setActive(i){
      [...ul.children].forEach((li,idx)=>li.classList.toggle('active',idx===i));
      active=i;
    }
    function accept(i){
      if(i<0||i>=items.length)return;
      const val=items[i].value;
      input.value=val;
      hide();
      const form=input.closest('form')||document.querySelector('#predictForm');
      if(form){
        const tpl=form.getAttribute('data-action-template')||form.getAttribute('action')||'';
        if(tpl.includes('TICK')) form.setAttribute('action', tpl.replace('TICK', encodeURIComponent(val)));
      }
    }

    const fetchSuggestions = debounce(q=>{
      if(!q){ hide(); return; }
      fetch(`/autocomplete/?q=${encodeURIComponent(q)}`)
        .then(r=>r.json())
        .then(render)
        .catch(()=>hide());
    },250);

    input.addEventListener('input', ()=>fetchSuggestions(input.value.trim()));
    input.addEventListener('keydown', e=>{
      if(!visible) return;
      if(e.key==='ArrowDown'){ e.preventDefault(); setActive(Math.min(active+1,items.length-1)); }
      else if(e.key==='ArrowUp'){ e.preventDefault(); setActive(Math.max(active-1,0)); }
      else if(e.key==='Enter'){ if(active>=0){ e.preventDefault(); accept(active); } }
      else if(e.key==='Escape'){ hide(); }
    });
    document.addEventListener('click', e=>{ if(!sugg.contains(e.target)&&e.target!==input) hide(); });
    window.addEventListener('resize', position);
    window.addEventListener('scroll', position,{passive:true});
  });
})();

