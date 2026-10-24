/* Sentinel Credit — shared page chrome: the mobile menu (full-screen list
 * below 900px). Same-origin, no dependencies. */
(() => {
  const btn = document.querySelector(".menu-btn");
  if (!btn) return;
  const set = open => {
    document.body.classList.toggle("menu-open", open);
    btn.setAttribute("aria-expanded", String(open));
    btn.setAttribute("aria-label", open ? "Close menu" : "Open menu");
  };
  btn.addEventListener("click", () => set(!document.body.classList.contains("menu-open")));
  document.querySelectorAll("#site-nav a").forEach(a => a.addEventListener("click", () => set(false)));
  document.addEventListener("keydown", e => { if (e.key === "Escape") set(false); });
  matchMedia("(min-width: 900px)").addEventListener("change", e => { if (e.matches) set(false); });
})();
