/* Renders proportional bar widths on the dashboard and analysis pages.
   The page works with this script disabled since the server already
   renders the numbers in the DOM as data attributes; this only sizes the
   bars visually. */

(function () {
  document.querySelectorAll("[data-bar-fill]").forEach((element) => {
    const value = parseFloat(element.getAttribute("data-bar-fill"));
    const max = parseFloat(element.getAttribute("data-bar-max") || "1");
    const percent = Math.max(0, Math.min(100, (value / max) * 100));
    element.style.width = `${percent}%`;
  });
})();
