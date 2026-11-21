// Context readers are rebuilt after navigation and progressive expansion.
// Delegate on the document rather than attaching listeners to transient rows.
(function () {
  "use strict";

  const unitSelector = "[data-geco-context-unit][data-geco-context-surface]";

  function navigate(element) {
    if (!element || !window.dash_clientside?.set_props) return;
    const unitId = Number(element.dataset.gecoContextUnit);
    const surface = element.dataset.gecoContextSurface;
    if (!Number.isSafeInteger(unitId) || !["explore", "focus"].includes(surface)) return;
    window.dash_clientside.set_props("context-navigation-request", {
      data: { unit_id: unitId, surface: surface, nonce: Date.now() },
    });
  }

  // The text is distinct from the adjacent span checkbox. A native dblclick
  // produces no checkbox mutations; do not cancel selection or scrolling.
  document.addEventListener("dblclick", function (event) {
    if (event.target.closest("input, label, button, a, textarea, select")) return;
    navigate(event.target.closest(unitSelector));
  });

  // Make the same destination accessible without a mouse, but do not
  // override typing/selection gestures or intercept descendant controls.
  document.addEventListener("keydown", function (event) {
    if (event.key !== "Enter" && event.key !== " ") return;
    const element = event.target.closest?.(unitSelector);
    if (!element || event.target !== element) return;
    if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
    event.preventDefault();
    navigate(element);
  });
})();
