function cloneWeights(weights) {
  return (weights || []).map((row) => [...row]);
}

function formatScaleValue(value) {
  if (Number.isInteger(value)) {
    return String(value);
  }
  return Number(value).toFixed(2).replace(/\.?0+$/, "");
}

function clampValue(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function chunkItems(items, chunkSize) {
  const chunks = [];
  for (let startIdx = 0; startIdx < items.length; startIdx += chunkSize) {
    chunks.push(items.slice(startIdx, startIdx + chunkSize));
  }
  return chunks;
}

function editorKey(target) {
  return `${target.dataset.channelIdx}:${target.dataset.featureIdx}`;
}

function parseEditableValue(rawValue) {
  const normalized = String(rawValue)
    .trim()
    .replace(/\u2212/g, "-")
    .replace(",", ".");

  if (
    normalized === "" ||
    normalized === "-" ||
    normalized === "." ||
    normalized === "-."
  ) {
    return null;
  }

  const value = Number(normalized);
  return Number.isNaN(value) ? NaN : value;
}

function render({ model, el }) {
  let isDragging = false;
  let activeEditor = null;

  const updateWeight = (featureIdx, channelIdx, rawValue) => {
    const sliderMin = Number(model.get("slider_min"));
    const sliderMax = Number(model.get("slider_max"));
    const value = clampValue(Number(rawValue), sliderMin, sliderMax);
    if (Number.isNaN(value)) {
      return;
    }

    const next = cloneWeights(model.get("weights"));
    next[channelIdx][featureIdx] = value;
    model.set("weights", next);
    model.save_changes();
  };

  const updateUI = () => {
    const maxCardsPerRow = 8;
    const title = model.get("title") || "Coefficient Editor";
    const subtitle = model.get("subtitle") || "";
    const features = model.get("features") || [];
    const channelLabels = model.get("channel_labels") || [];
    const weights = model.get("weights") || [];
    const sliderMin = model.get("slider_min");
    const sliderMax = model.get("slider_max");
    const sliderStep = model.get("slider_step");
    const sliderMid =
      sliderMin <= 0 && sliderMax >= 0 ? 0 : (sliderMin + sliderMax) / 2;

    if (!features.length || !weights.length) {
      el.innerHTML = `
        <div class="ce-shell">
          <div class="ce-empty">No editable coefficients for the selected state.</div>
        </div>
      `;
      return;
    }

    const legend = channelLabels
      .map(
        (label, idx) => `
          <span class="ce-legend-item">
            <span class="ce-legend-dot ce-dot-${idx % 3}"></span>
            ${label}
          </span>
        `,
      )
      .join("");

    const channelCards = features
      .map((feature, featureIdx) => {
        const sliders = weights
          .map((row, channelIdx) => {
            const value = Number(row[featureIdx] ?? 0);
            return `
              <div class="ce-slider-col">
                <input
                  type="number"
                  class="ce-value ce-value-${channelIdx % 3}"
                  min="${sliderMin}"
                  max="${sliderMax}"
                  step="${sliderStep}"
                  value="${value.toFixed(2)}"
                  data-feature-idx="${featureIdx}"
                  data-channel-idx="${channelIdx}"
                  aria-label="${feature} ${channelLabels[channelIdx] || `Channel ${channelIdx + 1}`} value"
                />
                <div class="ce-rail">
                  <input
                    type="range"
                    class="ce-slider ce-slider-${channelIdx % 3}"
                    orient="vertical"
                    min="${sliderMin}"
                    max="${sliderMax}"
                    step="${sliderStep}"
                    value="${value}"
                    data-feature-idx="${featureIdx}"
                    data-channel-idx="${channelIdx}"
                    aria-label="${feature} ${channelLabels[channelIdx] || `Channel ${channelIdx + 1}`}"
                  />
                </div>
                <div class="ce-channel-label">${channelLabels[channelIdx] || `C${channelIdx + 1}`}</div>
              </div>
            `;
          })
          .join("");

        return `
          <div class="ce-channel-card">
            <div class="ce-channel-header">${feature}</div>
            <div class="ce-slider-stack">
              <div class="ce-scale">
                <span class="ce-scale-mark">${formatScaleValue(sliderMax)}</span>
                <span class="ce-scale-mark">${formatScaleValue(sliderMid)}</span>
                <span class="ce-scale-mark">${formatScaleValue(sliderMin)}</span>
              </div>
              ${sliders}
            </div>
          </div>
        `;
      });

    const rows = chunkItems(channelCards, maxCardsPerRow)
      .map(
        (rowCards) => `
          <div class="ce-board-row">${rowCards.join("")}</div>
        `,
      )
      .join("");

    el.innerHTML = `
      <div class="ce-shell">
        <div class="ce-header">
          <div class="ce-heading">
            <div class="ce-title">${title}</div>
            <div class="ce-subtitle">${subtitle}</div>
          </div>
          <div class="ce-actions">
            <div class="ce-legend">${legend}</div>
            <button class="ce-reset" type="button">Reset</button>
          </div>
        </div>
        <div class="ce-board">${rows}</div>
      </div>
    `;

    const sliders = el.querySelectorAll(".ce-slider");
    sliders.forEach((slider) => {
      slider.addEventListener("pointerdown", () => {
        isDragging = true;
      });
      slider.addEventListener("input", (event) => {
        const featureIdx = Number(event.target.dataset.featureIdx);
        const channelIdx = Number(event.target.dataset.channelIdx);
        const value = Number(event.target.value);
        const valueNode = event.target.closest(".ce-slider-col").querySelector(".ce-value");
        if (valueNode) {
          valueNode.value = value.toFixed(2);
        }
        updateWeight(featureIdx, channelIdx, value);
      });
    });

    const valueInputs = el.querySelectorAll(".ce-value");
    valueInputs.forEach((input) => {
      const commitInput = (target) => {
        const featureIdx = Number(target.dataset.featureIdx);
        const channelIdx = Number(target.dataset.channelIdx);
        const slider = target.closest(".ce-slider-col").querySelector(".ce-slider");
        const parsedValue = parseEditableValue(target.value);
        if (parsedValue === null || Number.isNaN(parsedValue)) {
          target.value = Number(model.get("weights")[channelIdx][featureIdx] ?? 0).toFixed(2);
          return;
        }
        const value = clampValue(
          parsedValue,
          Number(model.get("slider_min")),
          Number(model.get("slider_max")),
        );
        if (slider) {
          slider.value = String(value);
        }
        updateWeight(featureIdx, channelIdx, value);
        target.value = value.toFixed(2);
      };

      input.addEventListener("focus", (event) => {
        activeEditor = editorKey(event.target);
      });

      input.addEventListener("input", (event) => {
        const target = event.target;
        const featureIdx = Number(target.dataset.featureIdx);
        const channelIdx = Number(target.dataset.channelIdx);
        const slider = target.closest(".ce-slider-col").querySelector(".ce-slider");
        const parsedValue = parseEditableValue(target.value);
        if (parsedValue === null || Number.isNaN(parsedValue)) {
          return;
        }
        const value = clampValue(
          parsedValue,
          Number(model.get("slider_min")),
          Number(model.get("slider_max")),
        );
        if (slider) {
          slider.value = String(value);
        }
        updateWeight(featureIdx, channelIdx, value);
      });

      input.addEventListener("change", (event) => {
        commitInput(event.target);
      });

      input.addEventListener("blur", (event) => {
        activeEditor = null;
        commitInput(event.target);
        updateUI();
      });

      input.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
          event.preventDefault();
          commitInput(event.target);
          event.target.blur();
        }
        if (event.key === "Escape") {
          event.preventDefault();
          const featureIdx = Number(event.target.dataset.featureIdx);
          const channelIdx = Number(event.target.dataset.channelIdx);
          event.target.value = Number(model.get("weights")[channelIdx][featureIdx] ?? 0).toFixed(2);
          activeEditor = null;
          event.target.blur();
        }
      });
    });

    const resetButton = el.querySelector(".ce-reset");
    if (resetButton) {
      resetButton.addEventListener("click", () => {
        model.set("weights", cloneWeights(model.get("original_weights")));
        model.save_changes();
      });
    }
  };

  const stopDragging = () => {
    if (!isDragging) {
      return;
    }
    isDragging = false;
    updateUI();
  };

  updateUI();
  document.addEventListener("pointerup", stopDragging);
  model.on("change:title", updateUI);
  model.on("change:subtitle", updateUI);
  model.on("change:features", updateUI);
  model.on("change:channel_labels", updateUI);
  model.on("change:original_weights", updateUI);
  model.on("change:weights", () => {
    if (!isDragging && activeEditor === null) {
      updateUI();
    }
  });
}

export default { render };
