(function() {
  'use strict';

  function applyTheme(theme) {
    const root = document.documentElement;
    if (theme === 'light' || theme === 'dark') {
      root.setAttribute('data-theme', theme);
    } else {
      root.removeAttribute('data-theme');
    }
  }

  function getStoredTheme() {
    try {
      return localStorage.getItem('preferred-theme');
    } catch {
      return null;
    }
  }

  function storeTheme(theme) {
    try {
      localStorage.setItem('preferred-theme', theme);
    } catch {
      // ignore write errors (storage might be disabled)
    }
  }

  function getCurrentTheme() {
    const forced = document.documentElement.getAttribute('data-theme');
    if (forced) return forced;
    const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    return prefersDark ? 'dark' : 'light';
  }

  function updateToggleButtonState(button, theme) {
    const isDark = theme === 'dark';
    button.setAttribute('aria-pressed', String(isDark));
    button.textContent = isDark ? 'Switch to light mode' : 'Switch to dark mode';
  }

  function handleToggleClick(event) {
    const button = event.currentTarget;
    const current = getCurrentTheme();
    const next = current === 'dark' ? 'light' : 'dark';
    applyTheme(next);
    storeTheme(next);
    updateToggleButtonState(button, next);
  }

  function init() {
    const savedTheme = getStoredTheme();
    if (savedTheme === 'light' || savedTheme === 'dark') {
      applyTheme(savedTheme);
    }

    const toggle = document.getElementById('theme-toggle');
    if (toggle) {
      updateToggleButtonState(toggle, getCurrentTheme());
      toggle.addEventListener('click', handleToggleClick);
    }

    const yearEl = document.getElementById('year');
    if (yearEl) {
      yearEl.textContent = String(new Date().getFullYear());
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
