function render({ model, el }) {
  el.innerHTML = "";

  const wrap = document.createElement("div");
  wrap.className = "save-figure-wrap";

  const button = document.createElement("button");
  button.className = "save-figure-btn";

  const applyTheme = () => {
    const theme = model.get("theme_tokens") || {};
    for (const [key, value] of Object.entries(theme)) {
      wrap.style.setProperty(`--save-figure-${key}`, value);
    }
  };

  const updateButton = () => {
    button.textContent = model.get("label");
    button.disabled = !!model.get("disabled");
  };

  button.addEventListener("click", () => {
    if (button.disabled) return;
    const clicks = model.get("clicks") || 0;
    model.set("clicks", clicks + 1);
    model.save_changes();
  });

  model.on("change:label", updateButton);
  model.on("change:disabled", updateButton);
  model.on("change:theme_tokens", applyTheme);

  applyTheme();
  updateButton();

  wrap.appendChild(button);
  el.appendChild(wrap);
}

export default { render };
