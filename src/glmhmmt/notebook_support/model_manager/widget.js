function escapeHTML(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function membersForGroup(group) {
  if (Array.isArray(group.toggle_members) && group.toggle_members.length > 0) {
    return [...group.toggle_members];
  }
  return Object.values(group.members || {});
}

function selectedState(selectedSet, members) {
  if (!members.length) {
    return "none";
  }
  const count = members.filter((member) => selectedSet.has(member)).length;
  if (count === 0) {
    return "none";
  }
  if (count === members.length) {
    return "all";
  }
  return "partial";
}

function renderSelectAll(subjectsList, currentSubjects) {
  const total = subjectsList.length;
  const count = currentSubjects.length;
  const state = total > 0 && count === total ? "selected" : count > 0 ? "partial" : "";
  const label = total > 0 && count === total ? "Deselect All" : "Select All";
  return `<button type="button" class="mm-select-all ${state}" id="btn-select-all">${label}</button>`;
}

function renderSubjectChips(subjectsList, currentSubjects) {
  return `
    <div class="mm-chip-container subjects-grid">
      ${subjectsList
        .map(
          (subject) => `
            <button
              type="button"
              class="mm-chip ${currentSubjects.includes(subject) ? "selected" : ""}"
              data-subject="${escapeHTML(subject)}"
            >${escapeHTML(subject)}</button>
          `,
        )
        .join("")}
    </div>
  `;
}

function renderGroupSelectors(groups, selectedCols, attrName) {
  if (!groups || groups.length === 0) {
    return '<p class="mm-empty-note">No regressors available.</p>';
  }

  const selectedSet = new Set(selectedCols || []);

  return `
    <div class="mm-group-grid">
      ${groups
        .map((group) => {
          const groupMembers = membersForGroup(group);
          const groupState = selectedState(selectedSet, groupMembers);
          const groupStateClass = groupState === "all" ? "selected" : groupState === "partial" ? "partial" : "";
          const groupMembersJSON = JSON.stringify(groupMembers).replace(/'/g, "&#39;");
          const hideMembers = Boolean(group.hide_members);
          const visibleSides = ["L", "C", "N", "R"];

          let pills = "";
          if (hideMembers) {
            pills = `
              <button
                type="button"
                class="mm-pill mm-pill-wide ${groupStateClass}"
                data-${attrName}-group="${escapeHTML(group.key)}"
                data-${attrName}-members='${groupMembersJSON}'
              >Toggle group</button>
            `;
          } else {
            pills = visibleSides
              .map((side) => {
                const member = group.members ? group.members[side] : null;
                if (!member) {
                  return "";
                }
                const active = selectedSet.has(member) ? "selected" : "";
                const sideLabel = side === "C" ? "C" : side === "N" ? "N" : side;
                return `
                  <button
                    type="button"
                    class="mm-pill ${active}"
                    data-${attrName}="${escapeHTML(member)}"
                  >${sideLabel}</button>
                `;
              })
              .join("");
          }

          return `
            <div class="mm-group-card ${groupStateClass}">
              <button
                type="button"
                class="mm-group-header ${groupStateClass}"
                data-${attrName}-group="${escapeHTML(group.key)}"
                data-${attrName}-members='${groupMembersJSON}'
              >
                <span class="mm-group-title">${escapeHTML(group.label)}</span>
                <span class="mm-group-state">${groupState === "all" ? "all" : groupState === "partial" ? "some" : "none"}</span>
              </button>
              <div class="mm-group-pills">${pills}</div>
            </div>
          `;
        })
        .join("")}
    </div>
  `;
}

function renderFrozenTable(numStates, selectedFeatures, frozenEmissions) {
  if (!selectedFeatures || selectedFeatures.length === 0) {
    return '<p class="mm-empty-note">Select emission regressors to freeze per state.</p>';
  }

  const frozen = frozenEmissions || {};
  const head = selectedFeatures
    .map(
      (feature) => `
        <th class="mm-freeze-th" title="${escapeHTML(feature)}">
          <span class="mm-freeze-head-text">${escapeHTML(feature)}</span>
        </th>
      `,
    )
    .join("");

  const rows = Array.from({ length: numStates }, (_, stateIdx) => {
    const stateKey = String(stateIdx);
    const stateFrozen = frozen[stateKey] || {};
    const cells = selectedFeatures
      .map((feature) => {
        const value = stateFrozen[feature];
        const display = Number.isFinite(value) ? String(value) : "";
        return `
          <td class="mm-freeze-td">
            <input
              type="number"
              step="any"
              class="mm-freeze-input"
              data-state="${escapeHTML(stateKey)}"
              data-feature="${escapeHTML(feature)}"
              value="${display}"
              placeholder="free"
            >
          </td>
        `;
      })
      .join("");

    return `
      <tr class="mm-freeze-row">
        <td class="mm-freeze-state">state ${stateIdx}</td>
        ${cells}
      </tr>
    `;
  }).join("");

  return `
    <div class="mm-freeze-table-wrap">
      <table class="mm-freeze-table">
        <thead>
          <tr>
            <th class="mm-freeze-th-state">State</th>
            ${head}
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  `;
}

function renderLoadList(existingInfo, existingVal, showTransitionRegressors) {
  const rows = existingInfo || [];
  if (!rows.length) {
    return '<p class="mm-empty-note">No saved models were found.</p>';
  }

  return `
    <div class="mm-load-list">
      ${rows
        .map((info) => {
          const selected = info.id === existingVal ? "selected" : "";
          const isDefault = info.id === "__default__";
          return `
            <div class="mm-load-card ${selected}" data-model="${escapeHTML(info.id)}">
              <div class="mm-load-main">
                <div class="mm-load-name">
                  ${escapeHTML(info.name)}
                  ${isDefault ? '<span class="mm-default-badge">default</span>' : ""}
                </div>
                <div class="mm-load-meta">
                  <span>${escapeHTML(info.subjects)} subjects</span>
                  <span>K ${escapeHTML(info.K)}</span>
                  <span>tau ${escapeHTML(info.tau)}</span>
                  <span>${escapeHTML(info.cv || "none")}</span>
                </div>
                <div class="mm-load-detail"><strong>Emission:</strong> ${escapeHTML(info.regressors || "")}</div>
                ${
                  showTransitionRegressors
                    ? `<div class="mm-load-detail"><strong>Transition:</strong> ${escapeHTML(info.transition_regressors || "")}</div>`
                    : ""
                }
              </div>
              ${
                isDefault
                  ? ""
                  : `<button type="button" class="mm-btn-delete-row" data-delete-model="${escapeHTML(info.id)}">Delete</button>`
              }
            </div>
          `;
        })
        .join("")}
    </div>
  `;
}

function render({ model, el }) {
  let aliasDraft = model.get("alias") || "";
  let aliasDirty = false;

  const sendCommand = (command, payload = {}) => {
    model.set("command", command);
    model.set("command_payload", payload);
    model.set("command_nonce", (model.get("command_nonce") || 0) + 1);
    model.save_changes();
  };

  const setTrait = (name, value) => {
    model.set(name, value);
    model.save_changes();
  };

  const toggleValues = (traitName, values) => {
    const current = new Set(model.get(traitName) || []);
    const allSelected = values.every((value) => current.has(value));
    if (allSelected) {
      values.forEach((value) => current.delete(value));
    } else {
      values.forEach((value) => current.add(value));
    }
    setTrait(traitName, [...current]);
  };

  const renderUI = () => {
    const taskOptions = model.get("task_options") || [];
    const taskDiscoveryMessage = model.get("task_discovery_message") || "";
    const currentTask = model.get("task");
    const mode = model.get("ui_mode");
    const modelType = model.get("model_type");
    const isRunning = model.get("is_running");
    const is2afc = model.get("is_2afc");
    const existingInfo = model.get("existing_models_info") || [];
    const existingVal = model.get("existing_model");
    const subjectsList = model.get("subjects_list") || [];
    const currentSubjects = model.get("subjects") || [];
    const currentK = model.get("K");
    const kOptions = model.get("k_options") || [2, 3, 4, 5, 6];
    const currentTau = model.get("tau");
    const currentCvMode = model.get("cv_mode") || "none";
    const currentCvRepeats = model.get("cv_repeats") || 5;
    const showConditionFilter = model.get("show_condition_filter");
    const conditionFilterOptions = model.get("condition_filter_options") || [];
    const currentConditionFilter = model.get("condition_filter") || "all";
    const currentLapseMode = model.get("lapse_mode") || "none";
    const currentLapseMax = model.get("lapse_max") || 0.2;
    const currentEmission = model.get("emission_cols") || [];
    const currentTransition = model.get("transition_cols") || [];
    const emissionGroups = model.get("emission_groups") || [];
    const transitionGroups = model.get("transition_groups") || [];
    const currentFrozen = model.get("frozen_emissions") || {};
    const alias = model.get("alias") || "";
    const aliasError = model.get("alias_error") || "";
    const aliasStatus = model.get("alias_status") || "";
    const savedModelName = model.get("saved_model_name") || "";

    if (!aliasDirty) {
      aliasDraft = alias;
    }

    const aliasMessage = aliasError || aliasStatus || (savedModelName ? `Current saved model: ${savedModelName}` : "");
    const aliasMessageClass = aliasError ? "error" : aliasStatus ? "status" : "hint";
    const showTransitionRegressors = modelType === "glmhmmt";

    if (!taskOptions.length) {
      el.innerHTML = `
        <div class="mm-content">
          <p class="mm-empty-note">${escapeHTML(taskDiscoveryMessage || "No task adapters were found.")}</p>
        </div>
      `;
      return;
    }

    let html = `
      <div class="mm-container">
        <div class="mm-header">
          <div class="mm-task-selector">
            <label class="mm-label inline">Task:</label>
            <select id="inp-task" class="mm-input small">
              ${taskOptions
                .map(
                  (opt) => `
                    <option value="${escapeHTML(opt.value)}" ${currentTask === opt.value ? "selected" : ""}>
                      ${escapeHTML(opt.label)}
                    </option>
                  `,
                )
                .join("")}
            </select>
          </div>
        </div>

        <div class="mm-tabs">
          <button type="button" class="mm-tab ${mode === "new" ? "active" : ""}" data-mode="new">New Fit</button>
          <button type="button" class="mm-tab ${mode === "load" ? "active" : ""}" data-mode="load">Load Existing</button>
        </div>

        <div class="mm-content">
    `;

    if (mode === "load") {
      html += `
        <div class="mm-section">
          <label class="mm-label">Saved Models</label>
          ${renderLoadList(existingInfo, existingVal, showTransitionRegressors)}
        </div>
      `;
    } else {
      html += `
        <div class="mm-flex-row">
          <div class="mm-col">
            <div class="mm-section">
              <label class="mm-label">Subjects</label>
              ${renderSelectAll(subjectsList, currentSubjects)}
              ${renderSubjectChips(subjectsList, currentSubjects)}
            </div>
      `;

      if (modelType !== "glm") {
        html += `
            <div class="mm-section">
              <label class="mm-label">Number of States (K)</label>
              <div class="mm-slider-wrap">
                <input
                  type="range"
                  class="mm-range"
                  id="inp-k-range"
                  min="${Math.min(...kOptions)}"
                  max="${Math.max(...kOptions)}"
                  step="1"
                  value="${currentK}"
                >
                <input
                  type="number"
                  class="mm-num-input"
                  id="inp-k-num"
                  min="${Math.min(...kOptions)}"
                  max="${Math.max(...kOptions)}"
                  step="1"
                  value="${currentK}"
                >
              </div>
            </div>
        `;
      }

      html += `
          </div>

          <div class="mm-col">
            <div class="mm-section">
              <label class="mm-label">Emission Regressors</label>
              ${renderGroupSelectors(emissionGroups, currentEmission, "emission")}
            </div>
      `;

      if (showTransitionRegressors) {
        html += `
            <div class="mm-section">
              <label class="mm-label">Transition Regressors</label>
              ${renderGroupSelectors(transitionGroups, currentTransition, "transition")}
            </div>
        `;
      }

      html += `
          </div>
        </div>
      `;

      if (modelType !== "glm") {
        html += `
          <div class="mm-section">
            <label class="mm-label">Frozen Emission Weights</label>
            ${renderFrozenTable(currentK, currentEmission, currentFrozen)}
          </div>
        `;
      }

      html += `
        <div class="mm-flex-row">
          <div class="mm-col half-col">
            <div class="mm-section">
              <label class="mm-label">Action Trace Half-life (tau)</label>
              <div class="mm-slider-wrap">
                <input type="range" class="mm-range" id="inp-tau-range" min="1" max="200" step="1" value="${currentTau}">
                <input type="number" class="mm-num-input" id="inp-tau-num" min="1" max="200" step="1" value="${currentTau}">
              </div>
            </div>
          </div>
      `;

      if (showConditionFilter) {
        html += `
          <div class="mm-col half-col">
            <div class="mm-section">
              <label class="mm-label">Condition Filter</label>
              <select id="inp-condition-filter" class="mm-input">
                ${conditionFilterOptions
                  .map(
                    (option) => `
                      <option value="${escapeHTML(option)}" ${currentConditionFilter === option ? "selected" : ""}>
                        ${escapeHTML(option)}
                      </option>
                    `,
                  )
                  .join("")}
              </select>
            </div>
          </div>
        `;
      }

      html += `</div>`;

      if (modelType !== "glm" && is2afc) {
        html += `
          <div class="mm-flex-row">
            <div class="mm-col half-col">
              <div class="mm-section">
                <label class="mm-label">Cross-validation Mode</label>
                <select id="inp-cv-mode" class="mm-input">
                  <option value="none" ${currentCvMode === "none" ? "selected" : ""}>none</option>
                  <option value="balanced_session_holdout" ${currentCvMode === "balanced_session_holdout" ? "selected" : ""}>balanced_session_holdout</option>
                </select>
              </div>
            </div>
            <div class="mm-col half-col">
              <div class="mm-section">
                <label class="mm-label">CV Repeats</label>
                <input
                  type="number"
                  class="mm-num-input"
                  id="inp-cv-repeats"
                  min="1"
                  step="1"
                  value="${currentCvRepeats}"
                  ${currentCvMode === "none" ? "disabled" : ""}
                >
              </div>
            </div>
          </div>
        `;
      }

      if (modelType === "glm") {
        html += `
          <div class="mm-flex-row">
            <div class="mm-col half-col">
              <div class="mm-section">
                <label class="mm-label">Lapse Mode</label>
                <select id="inp-lapse-mode" class="mm-input">
                  <option value="none" ${currentLapseMode === "none" ? "selected" : ""}>none</option>
                  <option value="class" ${currentLapseMode === "class" ? "selected" : ""}>class</option>
                  <option value="history" ${currentLapseMode === "history" ? "selected" : ""}>repeat/alternate (shared)</option>
                  <option value="history_conditioned" ${currentLapseMode === "history_conditioned" ? "selected" : ""}>repeat/alternate (conditioned)</option>
                </select>
              </div>
            </div>
            <div class="mm-col half-col">
              <div class="mm-section">
                <label class="mm-label">Max Lapse</label>
                <div class="mm-slider-wrap ${currentLapseMode === "none" ? "disabled" : ""}">
                  <input
                    type="range"
                    class="mm-range"
                    id="inp-lapse-max-range"
                    min="0.01"
                    max="1.0"
                    step="0.01"
                    value="${currentLapseMax}"
                    ${currentLapseMode === "none" ? "disabled" : ""}
                  >
                  <input
                    type="number"
                    class="mm-num-input"
                    id="inp-lapse-max-num"
                    min="0.01"
                    max="1.0"
                    step="0.01"
                    value="${currentLapseMax}"
                    ${currentLapseMode === "none" ? "disabled" : ""}
                  >
                </div>
              </div>
            </div>
          </div>
        `;
      }
    }

    html += `
      <hr class="mm-divider">
      <div class="mm-footer">
        <div class="mm-alias-wrap">
          <label class="mm-label inline">Custom Alias</label>
          <div class="mm-alias-controls">
            <input
              type="text"
              id="inp-alias"
              class="mm-input ${aliasError ? "error" : ""}"
              placeholder="e.g. my_best_fit"
            >
            <button type="button" class="mm-btn-secondary" id="btn-save-alias" ${isRunning ? "disabled" : ""}>Save</button>
          </div>
          <div class="mm-alias-message ${aliasMessageClass}">${escapeHTML(aliasMessage)}</div>
        </div>
        <button type="button" class="mm-btn-run" id="btn-run" ${isRunning ? "disabled" : ""}>
          ${isRunning ? "FITTING..." : "RUN FIT"}
        </button>
      </div>
    </div>
    `;

    el.innerHTML = html;

    const aliasInput = el.querySelector("#inp-alias");
    if (aliasInput) {
      aliasInput.value = aliasDraft;
    }

    const bind = (selector, eventName, handler) => {
      const node = el.querySelector(selector);
      if (node) {
        node.addEventListener(eventName, handler);
      }
    };

    const bindAll = (selector, eventName, handler) => {
      el.querySelectorAll(selector).forEach((node) => node.addEventListener(eventName, handler));
    };

    bindAll(".mm-tab", "click", (event) => {
      const modeValue = event.currentTarget.dataset.mode;
      setTrait("ui_mode", modeValue);
    });

    bind("#btn-select-all", "click", () => {
      const list = model.get("subjects_list") || [];
      const current = model.get("subjects") || [];
      setTrait("subjects", current.length === list.length ? [] : [...list]);
    });

    bindAll("[data-subject]", "click", (event) => {
      const subject = event.currentTarget.dataset.subject;
      const current = new Set(model.get("subjects") || []);
      if (current.has(subject)) {
        current.delete(subject);
      } else {
        current.add(subject);
      }
      setTrait("subjects", [...current]);
    });

    bindAll("[data-emission]", "click", (event) => {
      const value = event.currentTarget.dataset.emission;
      toggleValues("emission_cols", [value]);
    });

    bindAll("[data-transition]", "click", (event) => {
      const value = event.currentTarget.dataset.transition;
      toggleValues("transition_cols", [value]);
    });

    bindAll("[data-emission-group]", "click", (event) => {
      const raw = event.currentTarget.dataset.emissionMembers || "[]";
      toggleValues("emission_cols", JSON.parse(raw));
    });

    bindAll("[data-transition-group]", "click", (event) => {
      const raw = event.currentTarget.dataset.transitionMembers || "[]";
      toggleValues("transition_cols", JSON.parse(raw));
    });

    bind("#inp-task", "change", (event) => {
      setTrait("task", event.target.value);
    });

    const syncNumericPair = (rangeId, numberId, traitName, parseFn) => {
      bind(`#${rangeId}`, "input", (event) => {
        const value = parseFn(event.target.value);
        const other = el.querySelector(`#${numberId}`);
        if (other) {
          other.value = String(value);
        }
        setTrait(traitName, value);
      });

      bind(`#${numberId}`, "change", (event) => {
        const value = parseFn(event.target.value);
        const other = el.querySelector(`#${rangeId}`);
        if (other) {
          other.value = String(value);
        }
        setTrait(traitName, value);
      });
    };

    syncNumericPair("inp-k-range", "inp-k-num", "K", (value) => parseInt(value, 10));
    syncNumericPair("inp-tau-range", "inp-tau-num", "tau", (value) => parseInt(value, 10));
    syncNumericPair("inp-lapse-max-range", "inp-lapse-max-num", "lapse_max", (value) => parseFloat(value));

    bind("#inp-cv-mode", "change", (event) => {
      setTrait("cv_mode", event.target.value);
    });

    bind("#inp-cv-repeats", "change", (event) => {
      const value = parseInt(event.target.value, 10);
      setTrait("cv_repeats", Number.isFinite(value) && value > 0 ? value : 1);
    });

    bind("#inp-condition-filter", "change", (event) => {
      setTrait("condition_filter", event.target.value);
    });

    bind("#inp-lapse-mode", "change", (event) => {
      setTrait("lapse_mode", event.target.value);
    });

    const commitFreezeInput = (input) => {
      if (!input) {
        return;
      }
      const state = input.dataset.state;
      const feature = input.dataset.feature;
      if (!state || !feature) {
        return;
      }

      const current = JSON.parse(JSON.stringify(model.get("frozen_emissions") || {}));
      const rawValue = String(input.value || "").trim();

      if (!rawValue) {
        if (current[state]) {
          delete current[state][feature];
          if (Object.keys(current[state]).length === 0) {
            delete current[state];
          }
        }
      } else {
        const parsed = parseFloat(rawValue);
        if (!Number.isFinite(parsed)) {
          return;
        }
        if (!current[state]) {
          current[state] = {};
        }
        current[state][feature] = parsed;
      }

      setTrait("frozen_emissions", current);
    };

    bindAll(".mm-freeze-input", "change", (event) => {
      commitFreezeInput(event.target);
    });

    bindAll(".mm-freeze-input", "keydown", (event) => {
      if (event.key !== "Enter") {
        return;
      }
      event.preventDefault();
      commitFreezeInput(event.target);
      event.target.blur();
    });

    bindAll("[data-model]", "click", (event) => {
      if (event.target.closest(".mm-btn-delete-row")) {
        return;
      }
      const modelId = event.currentTarget.dataset.model;
      setTrait("existing_model", modelId);
    });

    bindAll(".mm-btn-delete-row", "click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      const name = event.currentTarget.dataset.deleteModel;
      if (!name) {
        return;
      }
      if (!window.confirm(`Delete model "${name}" and its folder?`)) {
        return;
      }
      sendCommand("delete_model", { name });
    });

    bind("#inp-alias", "input", (event) => {
      aliasDraft = event.target.value;
      aliasDirty = true;
    });

    bind("#inp-alias", "change", (event) => {
      aliasDraft = event.target.value;
      aliasDirty = false;
      setTrait("alias", aliasDraft);
    });

    bind("#inp-alias", "keydown", (event) => {
      if (event.key !== "Enter") {
        return;
      }
      event.preventDefault();
      aliasDraft = event.target.value;
      aliasDirty = false;
      sendCommand("save_alias", { alias: aliasDraft });
    });

    bind("#btn-save-alias", "click", () => {
      aliasDirty = false;
      sendCommand("save_alias", { alias: aliasDraft });
    });

    bind("#btn-run", "click", () => {
      if (model.get("is_running")) {
        return;
      }
      sendCommand("run_fit");
    });
  };

  renderUI();
  model.on("change", renderUI);

  return () => {
    model.off("change", renderUI);
  };
}

export default { render };
