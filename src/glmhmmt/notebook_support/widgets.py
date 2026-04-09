import anywidget
import traitlets
import string
import random

from .model_manager import ModelManagerWidget, model_cfg  # noqa: F401  (re-exported)

def random_string(length=8):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


class CoefTweakerWidget(anywidget.AnyWidget):
    _esm = """
    function render({ model, el }) {
      
      const updateUI = () => {
        const existingModels = model.get("existing_models");
        const existingVal = model.get("existing_model");
        
        const is2afc = model.get("is_2afc");
        const modelType = model.get("model_type"); // 'glm', 'glmhmm', 'glmhmmt'
        const currentTask = model.get("task");
        const taskOptions = model.get("task_options") || [];
        
        // Tab state: "load" or "new"
        const mode = model.get("ui_mode");
        
        // New configuration options
        const KList = model.get("k_options");
        const currentK = model.get("K");
        
        const subjectsList = model.get("subjects_list");
        const currentSubjects = model.get("subjects");
        
        const emissionOpts = model.get("emission_cols_options");
        const currentEmission = model.get("emission_cols");

        const transitionOpts = model.get("transition_cols_options");
        const currentTransition = model.get("transition_cols");
        
        // Shared configuration options
        const currentTau = model.get("tau");
        const currentLapse = model.get("lapse");
        const currentLapseMax = model.get("lapse_max");
        const currentAlias = model.get("alias");

        const existingModelsInfo = model.get("existing_models_info");

        // Styling the table and shrinking the tau column:
        let html = `
          <div class="mm-container" id="${containerId}">
            <div class="mm-header">
                <div class="mm-task-selector">
                    <label class="mm-label inline">Task:</label>
                    <select id="inp-task" class="mm-input small">
                        ${taskOptions.map(opt => `
                            <option value="${opt.value}" ${currentTask === opt.value ? "selected" : ""}>${opt.label}</option>
                        `).join("")}
                    </select>
                </div>
            </div>
            <div class="mm-tabs">
              <button class="mm-tab ${mode === 'new' ? 'active' : ''}" data-mode="new">New Fit</button>
              <button class="mm-tab ${mode === 'load' ? 'active' : ''}" data-mode="load" ${existingModelsInfo.length===0?'disabled':''}>Load Existing</button>
            </div>
            
            <div class="mm-content">
        `;
        
        if (mode === 'load') {
            html += `
                <div class="mm-section">
                    <label class="mm-label">Select Saved Model</label>
                    <div class="mm-table-container">
                        <table class="mm-table">
                            <thead>
                                <tr>
                                    <th>Model Name</th>
                                    <th>Subjects</th>
                                    <th>K</th>
                                    <th>Regressors</th>
                                    <th>Tau</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${existingModelsInfo.map((info) => `
                                    <tr class="mm-tr ${info.name === existingVal ? 'selected' : ''}" data-model="${info.name}">
                                        <td><strong>${info.name}</strong></td>
                                        <td>${info.subjects}</td>
                                        <td>${info.K}</td>
                                        <td><div class="mm-truncate" title="${info.regressors}">${info.regressors}</div></td>
                                        <td>${info.tau}</td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                </div>
            `;
        } else {
            html += `
                <!-- NEW FIT CONFIGURATION -->
                <div class="mm-flex-row">
                    <!-- Column 1: Subjects & K -->
                    <div class="mm-col">
                        <div class="mm-section">
                            <label class="mm-label">Subjects</label>
                            <div class="mm-chip-container subjects-grid">
                            ${subjectsList.map(s => `
                                <div class="mm-chip ${currentSubjects.includes(s) ? 'selected' : ''}" data-subject="${s}">
                                    ${s}
                                </div>
                            `).join('')}
                            </div>
                        </div>
            `;
            
            if (modelType !== 'glm') {
                html += `
                        <div class="mm-section">
                            <label class="mm-label">Number of States (K)</label>
                            <div class="mm-slider-wrap">
                                <input type="range" class="mm-range" id="inp-k-range" min="${Math.min(...KList)}" max="${Math.max(...KList)}" value="${currentK}" step="1">
                                <input type="number" class="mm-num-input" id="inp-k-num" min="${Math.min(...KList)}" max="${Math.max(...KList)}" value="${currentK}" step="1">
                            </div>
                        </div>
                `;
            }

            html += `
                    </div>
                    
                    <!-- Column 2: Regressors -->
                    <div class="mm-col">
                        <div class="mm-section">
                            <label class="mm-label">Emission Regressors</label>
                            <div class="mm-chip-container regressors-grid">
                            ${emissionOpts.map(e => `
                                <div class="mm-chip ${currentEmission.includes(e) ? 'selected' : ''}" data-emission="${e}">
                                    ${e}
                                </div>
                            `).join('')}
                            </div>
                        </div>
            `;
            
            if (modelType === 'glmhmmt') {
                 html += `
                        <div class="mm-section">
                            <label class="mm-label">Transition Regressors</label>
                            <div class="mm-chip-container transition-grid">
                            ${transitionOpts.map(e => `
                                <div class="mm-chip trans-chip ${currentTransition.includes(e) ? 'selected' : ''}" data-transition="${e}">
                                    ${e}
                                </div>
                            `).join('')}
                            </div>
                        </div>
                 `;
            }

            html += `
                    </div>
                </div>
                
                <hr class="mm-divider"/>

                <div class="mm-flex-row">
                    <!-- Tau Half Screen Column -->
                    <div class="mm-col half-col">
                         <div class="mm-section">
                            <label class="mm-label">Action Trace Half-life (τ)</label>
                            <div class="mm-slider-wrap">
                                <input type="range" class="mm-range" id="inp-tau-range" min="1" max="200" value="${currentTau}" step="1">
                                <input type="number" class="mm-num-input" id="inp-tau-num" min="1" max="200" value="${currentTau}" step="1">
                            </div>
                        </div>
                    </div>
            `;

            if (modelType === 'glm' && is2afc) {
                html += `
                    <div class="mm-col">
                         <div class="mm-section row-align">
                            <label class="mm-checkbox">
                                <input type="checkbox" id="inp-lapse" ${currentLapse ? 'checked':''}>
                                <span class="mm-checkmark"></span>
                                Fit Lapse Rates
                            </label>
                            <div class="mm-slider-wrap tight ${!currentLapse ? 'disabled':''}">
                                <span class="mm-label inline">Max Lapse:</span>
                                <input type="range" class="mm-range" id="inp-lapse-max-range" min="0.01" max="1.0" value="${currentLapseMax}" step="0.01" ${!currentLapse ? 'disabled':''}>
                                <input type="number" class="mm-num-input" id="inp-lapse-max-num" min="0.01" max="1.0" value="${currentLapseMax}" step="0.01" ${!currentLapse ? 'disabled':''}>
                            </div>
                        </div>
                    </div>
                `;
            }

            html += `</div>`; // end flex row
        } // end new fit block

        html += `
                <hr class="mm-divider"/>
                <div class="mm-footer">
                    <div class="mm-alias-wrap">
                        <label class="mm-label inline">Custom Alias:</label>
                        <input type="text" id="inp-alias" class="mm-input" placeholder="e.g. my_best_fit" value="${currentAlias}">
                        <button class="mm-btn-secondary" id="btn-save-alias">Save</button>
                    </div>
                    <button class="mm-btn-run" id="btn-run">RUN FIT</button>
                </div>
            </div> <!-- mm-content -->
          </div> <!-- mm-container -->
        `;
        
        el.innerHTML = html;

        // --- Event Listeners ---
        const bind = (sel, ev, handler) => {
            const node = el.querySelector(sel);
            if (node) node.addEventListener(ev, handler);
        };
        const bindAll = (sel, ev, handler) => {
            const nodes = el.querySelectorAll(sel);
            nodes.forEach(n => n.addEventListener(ev, handler));
        };

        // Tabs
        bindAll(".mm-tab", "click", (e) => {
            model.set("ui_mode", e.target.dataset.mode);
            // Clear existing model if switching to new
            if (e.target.dataset.mode === 'new') model.set("existing_model", "");
            model.save_changes();
        });

        // Load existing - Table Row Click
        bindAll(".mm-tr", "click", (e) => {
            const row = e.target.closest(".mm-tr");
            if(row) {
                // remove selected from others
                el.querySelectorAll(".mm-tr").forEach(r => r.classList.remove("selected"));
                row.classList.add("selected");
                model.set("existing_model", row.dataset.model);
                model.save_changes();
            }
        });

        // Chips - Subjects
        bindAll(".mm-chip[data-subject]", "click", (e) => {
            const sub = e.target.dataset.subject;
            let current = [...model.get("subjects")];
            if (current.includes(sub)) {
                current = current.filter(x => x !== sub);
            } else {
                current.push(sub);
            }
            model.set("subjects", current);
            model.save_changes();
        });

        // Chips - Emission
        bindAll(".mm-chip[data-emission]", "click", (e) => {
            const em = e.target.dataset.emission;
            let current = [...model.get("emission_cols")];
            if (current.includes(em)) {
                current = current.filter(x => x !== em);
            } else {
                current.push(em);
            }
            model.set("emission_cols", current);
            model.save_changes();
        });

        // Chips - Transition
        bindAll(".mm-chip[data-transition]", "click", (e) => {
            const tr = e.target.dataset.transition;
            let current = [...model.get("transition_cols")];
            if (current.includes(tr)) {
                current = current.filter(x => x !== tr);
            } else {
                current.push(tr);
            }
            model.set("transition_cols", current);
            model.save_changes();
        });

        bind("#inp-task", "change", (e) => {
            model.set("task", e.target.value);
            model.save_changes();
        });

        // Sliders synchronized with number inputs
        const syncSliderAndNum = (rangeId, numId, traitName, parseFn) => {
            bind("#" + rangeId, "input", (e) => {
                const val = parseFn(e.target.value);
                const numEl = el.querySelector("#" + numId);
                if (numEl) numEl.value = val;
                model.set(traitName, val);
                model.save_changes();
            });
            bind("#" + numId, "change", (e) => {
                let val = parseFn(e.target.value);
                // HTML5 handles min/max constraints on valid form submit, but we enforce loosely here if needed.
                const rangeEl = el.querySelector("#" + rangeId);
                if (rangeEl) rangeEl.value = val;
                model.set(traitName, val);
                model.save_changes();
            });
        };

        syncSliderAndNum("inp-k-range", "inp-k-num", "K", parseInt);
        syncSliderAndNum("inp-tau-range", "inp-tau-num", "tau", parseInt);
        syncSliderAndNum("inp-lapse-max-range", "inp-lapse-max-num", "lapse_max", parseFloat);

        // Checkbox
        bind("#inp-lapse", "change", (e) => {
            model.set("lapse", e.target.checked);
            model.save_changes();
        });

        // Alias
        bind("#inp-alias", "input", (e) => {
            model.set("alias", e.target.value);
            model.save_changes();
        });
        
        bind("#btn-save-alias", "click", (e) => {
            model.set("save_alias_clicks", model.get("save_alias_clicks") + 1);
            model.save_changes();
            const btn = e.target;
            const originalText = btn.innerText;
            btn.innerText = "Saved!";
            btn.classList.add("saved");
            setTimeout(() => {
                btn.innerText = originalText;
                btn.classList.remove("saved");
            }, 1000);
        });

        // Run
        bind("#btn-run", "click", (e) => {
            model.set("run_fit_clicks", model.get("run_fit_clicks") + 1);
            model.save_changes();
            
            // Temporary button animation
            const btn = e.target;
            const originalText = btn.innerText;
            btn.innerText = "FITTING...";
            btn.classList.add("running");
            setTimeout(() => {
                btn.innerText = originalText;
                btn.classList.remove("running");
            }, 800);
        });
      };

      updateUI();
      model.on("change", updateUI);
    }
    export default { render };
    """
    
    _css = """
    :host {
        --brand-main: #003660;
        --brand-light: #004b87;
        --brand-dark: #002240;
        --brand-bg: #e6f0f9;
        font-family: system-ui, -apple-system, sans-serif;
        background: #ffffff;
        border: 2px solid #e5e7eb;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        overflow: hidden;
        color: #000000;
        width: 100%;
        max-width: 100%;
        margin: 10px 0;
    }

    .mm-header {
        padding: 12px 20px;
        background: #ffffff;
        border-bottom: 2px solid #e5e7eb;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .mm-task-selector {
        display: flex;
        align-items: center;
        background: #f3f4f6;
        padding: 4px 12px;
        border-radius: 8px;
    }
    
    .mm-input.small {
        padding: 4px 8px;
        border-radius: 4px;
        border: none;
        background: transparent;
        font-weight: 600;
        cursor: pointer;
        font-size: 15px;
    }

    .mm-tabs {
        display: flex;
        background: #f9fafb;
        border-bottom: 2px solid #e5e7eb;
    }
    
    .mm-tab {
        flex: 1;
        padding: 16px 16px;
        background: none;
        border: none;
        border-bottom: 3px solid transparent;
        font-weight: 700;
        font-size: 16px;
        color: #4b5563;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .mm-tab:hover:not(:disabled) {
        color: #000000;
        background: #e5e7eb;
    }
    
    .mm-tab.active {
        color: var(--brand-main);
        border-bottom-color: var(--brand-main);
        background: #ffffff;
    }
    
    .mm-tab:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }
    
    .mm-content {
        padding: 20px;
    }
    
    .mm-flex-row {
        display: flex;
        flex-wrap: wrap;
        gap: 24px;
        margin-bottom: 16px;
    }
    
    .mm-col.half-col {
        flex: 0 0 calc(50% - 12px);
    }
    
    .mm-section {
        margin-bottom: 20px;
    }
    
    .mm-label {
        display: block;
        font-weight: 800;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #374151;
        margin-bottom: 10px;
    }
    
    .mm-label.inline {
        display: inline-block;
        margin-bottom: 0;
        margin-right: 8px;
    }
    
    /* Chips */
    .mm-chip-container {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
    }
    
    .mm-chip {
        padding: 8px 14px;
        border-radius: 10px;
        font-size: 15px;
        font-weight: 600;
        cursor: pointer;
        background: #ffffff;
        color: #9ca3af;
        border: 2px solid #9ca3af;
        transition: all 0.15s ease;
        user-select: none;
    }
    
    .mm-chip:hover {
        background: #f3f4f6;
        border-color: #4b5563;
    }
    
    .mm-chip.selected {
        background: var(--brand-bg);
        color: var(--brand-main);
        border-color: var(--brand-light);
        box-shadow: 0 1px 2px rgba(0, 54, 96, 0.1);
    }

    .mm-chip.trans-chip.selected {
        background: #ccfbf1;
        color: #115e59;
        border-color: #14b8a6;
    }
    
    .mm-input {
        width: 100%;
        padding: 10px 14px;
        border-radius: 6px;
        border: 2px solid #d1d5db;
        font-size: 16px;
        background: #ffffff;
        color: #000000;
        outline: none;
        transition: border-color 0.2s;
        box-sizing: border-box;
    }
    
    .mm-input:focus {
        border-color: var(--brand-light);
        box-shadow: 0 0 0 2px rgba(0, 75, 135, 0.2);
    }
    
    .mm-num-input {
        width: 70px;
        padding: 8px 10px;
        border-radius: 4px;
        border: 2px solid #d1d5db;
        font-family: monospace;
        text-align: center;
        font-size: 16px;
        background: #ffffff;
        color: #000000;
        font-weight: 700;
    }
    
    .mm-num-input:focus {
        outline: none;
        border-color: var(--brand-light);
    }

    /* Sliders */
    .mm-slider-wrap {
        display: flex;
        align-items: center;
        gap: 12px;
        background: #ffffff;
        padding: 10px 14px;
        border-radius: 8px;
        border: 2px solid #e5e7eb;
    }
    
    .mm-slider-wrap.tight {
        padding: 4px 8px;
        background: transparent;
        border: none;
    }

    .mm-slider-wrap.disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }
    
    .mm-slider-wrap input[type=range] {
        flex: 1;
        accent-color: var(--brand-light);
    }
    
    /* Checkbox */
    .row-align {
        display: flex;
        align-items: center;
        gap: 16px;
    }
    
    /* Dividers */
    .mm-divider {
        border: none;
        border-top: 2px dashed #d1d5db;
        margin: 24px 0;
    }
    
    /* Footer */
    .mm-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 16px;
        flex-wrap: wrap;
    }
    
    .mm-alias-wrap {
        flex: 1;
        min-width: 200px;
        display: flex;
        align-items: center;
    }
    
    .mm-alias-wrap .mm-input {
        width: 100%;
        max-width: 200px;
    }

    .mm-btn-secondary {
        background: #ffffff;
        color: #000000;
        border: 2px solid #d1d5db;
        padding: 12px 20px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 16px;
        cursor: pointer;
        transition: all 0.2s;
        margin-left: 8px;
    }

    .mm-btn-secondary:hover {
        background: #f3f4f6;
        border-color: #9ca3af;
    }

    .mm-btn-secondary.saved {
        background: #d1fae5;
        color: #047857;
        border-color: #6ee7b7;
    }
    
    .mm-btn-run {
        background: var(--brand-main);
        color: white;
        border: none;
        padding: 14px 32px;
        border-radius: 8px;
        font-weight: 800;
        font-size: 18px;
        letter-spacing: 0.1em;
        cursor: pointer;
        box-shadow: 0 4px 6px -1px rgba(0, 54, 96, 0.3);
        transition: all 0.2s;
        text-wrap: nowrap;
    }
    
    .mm-btn-run:hover {
        background: var(--brand-light);
        transform: translateY(-1px);
        box-shadow: 0 6px 8px -1px rgba(0, 54, 96, 0.4);
    }
    
    .mm-btn-run:active {
        transform: translateY(1px);
        box-shadow: 0 2px 4px -1px rgba(0, 54, 96, 0.3);
    }

    .mm-btn-run.running {
        background: linear-gradient(135deg, var(--emerald-5, #10b981), var(--emerald-6, #059669));
        cursor: wait;
        pointer-events: none;
    }
    
    /* Table Styles */
    .mm-table-container {
        max-height: 500px;
        overflow-y: auto;
        border: 2px solid #e5e7eb;
        border-radius: 8px;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);
    }
    
    .mm-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 16px;
        text-align: left;
    }
    
    .mm-table thead {
        background: #f3f4f6;
        position: sticky;
        top: 0;
        z-index: 10;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    
    .mm-table th {
        padding: 14px 18px;
        font-weight: 800;
        color: #111827;
        text-transform: uppercase;
        font-size: 14px;
        letter-spacing: 0.05em;
        border-bottom: 2px solid #d1d5db;
    }
    
    .mm-table td {
        padding: 14px 18px;
        border-bottom: 1px solid #e5e7eb;
        color: #000000;
        vertical-align: middle;
    }
    
    .mm-tr {
        cursor: pointer;
        transition: background 0.15s ease;
    }
    
    .mm-tr:hover {
        background: #f9fafb;
    }
    
    .mm-tr.selected {
        background: var(--brand-bg);
    }
    
    .mm-tr.selected td {
        color: var(--brand-dark);
        border-bottom-color: var(--brand-light);
        font-weight: 700;
    }
    
    .mm-truncate {
        max-width: 300px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* Dark Mode Support */
    @media (prefers-color-scheme: dark) {
        .mm-container {
            background: var(--gray-9, #111827);
            border-color: var(--gray-8, #1f2937);
            color: var(--gray-1, #f9fafb);
        }
        .mm-tabs {
            background: var(--gray-9, #111827);
            border-color: var(--gray-8, #1f2937);
        }
        .mm-tab.active {
            background: var(--gray-8, #1f2937);
            color: var(--blue-4, #60a5fa);
            border-bottom-color: var(--blue-4, #60a5fa);
        }
        .mm-tab {
            color: var(--gray-4, #9ca3af);
        }
        .mm-tab:hover:not(:disabled) {
            color: var(--gray-2, #e5e7eb);
            background: var(--gray-8, #1f2937);
        }
        .mm-chip {
            background: var(--gray-8, #1f2937);
            color: var(--gray-3, #d1d5db);
            border-color: var(--gray-7, #374151);
        }
        .mm-chip:hover {
            background: var(--gray-7, #374151);
        }
        .mm-chip.selected {
            background: rgba(59, 130, 246, 0.2);
            color: var(--blue-3, #93c5fd);
            border-color: var(--blue-5, #3b82f6);
        }
        .mm-chip.trans-chip.selected {
            background: rgba(20, 184, 166, 0.2);
            color: var(--teal-3, #5eead4);
            border-color: var(--teal-5, #14b8a6);
        }
        .mm-input {
            background: var(--gray-8, #1f2937);
            border-color: var(--gray-6, #4b5563);
        }
        .mm-num-input {
            background: var(--gray-9, #111827);
            border-color: var(--gray-6, #4b5563);
            color: var(--gray-2, #e5e7eb);
        }
        .mm-btn-secondary {
            background: var(--gray-8, #1f2937);
            color: var(--gray-3, #d1d5db);
            border-color: var(--gray-6, #4b5563);
        }
        .mm-btn-secondary:hover {
            background: var(--gray-7, #374151);
        }
        .mm-slider-wrap {
            background: var(--gray-8, #1f2937);
            border-color: var(--gray-7, #374151);
        }
        .mm-divider {
            border-color: var(--gray-7, #374151);
        }
    }
    """
    
    ui_mode = traitlets.Unicode("new").tag(sync=True) # "new" or "load"
    
    model_type = traitlets.Unicode("glmhmm").tag(sync=True) # "glm", "glmhmm", "glmhmmt"
    task = traitlets.Unicode("MCDR").tag(sync=True)
    is_2afc = traitlets.Bool(False).tag(sync=True)
    
    existing_models = traitlets.List(traitlets.Unicode()).tag(sync=True)
    existing_models_info = traitlets.List(traitlets.Dict()).tag(sync=True)
    existing_model = traitlets.Unicode("").tag(sync=True)
    
    alias = traitlets.Unicode("").tag(sync=True)
    
    subjects_list = traitlets.List(traitlets.Unicode()).tag(sync=True)
    subjects = traitlets.List(traitlets.Unicode()).tag(sync=True)
    
    k_options = traitlets.List(traitlets.Int(), default_value=[2,3,4,5,6]).tag(sync=True)
    K = traitlets.Int(2).tag(sync=True)
    
    tau = traitlets.Int(50).tag(sync=True)
    
    lapse = traitlets.Bool(False).tag(sync=True)
    lapse_max = traitlets.Float(0.2).tag(sync=True)
    
    emission_cols_options = traitlets.List(traitlets.Unicode()).tag(sync=True)
    emission_cols = traitlets.List(traitlets.Unicode()).tag(sync=True)
    
    transition_cols_options = traitlets.List(traitlets.Unicode()).tag(sync=True)
    transition_cols = traitlets.List(traitlets.Unicode()).tag(sync=True)

    run_fit_clicks = traitlets.Int(0).tag(sync=True)
    save_alias_clicks = traitlets.Int(0).tag(sync=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._update_options()

    @traitlets.observe("task", "model_type")
    def _on_task_change(self, change):
        self._update_options()

    def _update_options(self):
        fits_path = paths.RESULTS / "fits" / self.task / self.model_type
        opts = []
        info_list = []
        import json
        
        if fits_path.exists():
            for d in fits_path.iterdir():
                if d.is_dir() and (d / "config.json").exists():
                    v_name = d.name
                    opts.append(v_name)
                    # parse info for the table
                    try:
                        cfg = json.loads((d / "config.json").read_text())
                        info_dict = {
                            "name": v_name,
                            "subjects": len(cfg.get("subjects", [])),
                            "K": cfg.get("K") or cfg.get("K_list", [0])[0] if "K_list" in cfg else cfg.get("K", 0),
                            "tau": cfg.get("tau", ""),
                            "regressors": ", ".join(cfg.get("emission_cols", [])),
                        }
                        info_list.append(info_dict)
                    except Exception:
                        info_list.append({"name": v_name, "subjects": "?", "K": "?", "tau": "?", "regressors": "?"})
        
        # Sort by name
        opts = sorted(opts)
        info_list = sorted(info_list, key=lambda x: x["name"])
        
        self.existing_models = opts
        self.existing_models_info = info_list
        if self.existing_model not in self.existing_models:
            self.existing_model = ""

        try:
            adapter = get_adapter(self.task)
            self.is_2afc = adapter.num_classes == 2
            
            df_all = pl.read_parquet(paths.DATA_PATH / adapter.data_file)
            df_all = adapter.subject_filter(df_all)
            
            subjects = sorted(df_all["subject"].unique().to_list(), key=str)
            self.subjects_list = subjects
            if not self.subjects:
                self.subjects = subjects
                
            available_ecols = adapter.available_emission_cols(df_all)
            default_ecols = adapter.default_emission_cols(df_all)
            self.emission_cols_options = available_ecols
            if not self.emission_cols:
                self.emission_cols = default_ecols[:10] if self.model_type == 'glm' else default_ecols
                
            tcols = adapter.default_transition_cols()
            self.transition_cols_options = adapter.available_transition_cols()
            if not self.transition_cols:
                self.transition_cols = tcols
                
        except Exception as e:
            print(f"Error loading options for task {self.task}: {e}")


class CoefTweakerWidget(anywidget.AnyWidget):
    _esm = """
    function render({ model, el }) {
        const updateUI = () => {
            const features = model.get("features");
            const wL = model.get("w_L");
            const wR = model.get("w_R");
            
            if (features.length === 0) return;

            let html = `
                <div class="eq-container">
                    <div class="eq-header">
                        <div class="eq-title">Coefficient Equalizer (A92)</div>
                        <div class="eq-legend">
                            <span class="eq-dot dot-l"></span> W_L (Left)
                            <span class="eq-dot dot-r"></span> W_R (Right)
                        </div>
                    </div>
                    
                    <div class="eq-board">
                        ${features.map((feat, i) => `
                            <div class="eq-channel">
                                <div class="eq-sliders">
                                    <div class="eq-slider-col l-col">
                                        <div class="eq-val val-l">${wL[i].toFixed(2)}</div>
                                        <input type="range" class="eq-vertical l-input" orient="vertical" 
                                            min="-10" max="10" step="0.05" value="${wL[i]}" data-idx="${i}">
                                    </div>
                                    <div class="eq-slider-col r-col">
                                        <div class="eq-val val-r">${wR[i].toFixed(2)}</div>
                                        <input type="range" class="eq-vertical r-input" orient="vertical" 
                                            min="-10" max="10" step="0.05" value="${wR[i]}" data-idx="${i}">
                                    </div>
                                </div>
                                <div class="eq-label" title="${feat}">${feat}</div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `;
            
            el.innerHTML = html;

            const bindAll = (sel, ev, handler) => {
                const nodes = el.querySelectorAll(sel);
                nodes.forEach(n => n.addEventListener(ev, handler));
            };

            bindAll(".l-input", "input", (e) => {
                const idx = parseInt(e.target.dataset.idx);
                const val = parseFloat(e.target.value);
                let current = [...model.get("w_L")];
                current[idx] = val;
                model.set("w_L", current);
                // Also update the UI immediately without full re-render for smooth sliding
                e.target.previousElementSibling.innerText = val.toFixed(2);
                model.save_changes();
            });

            bindAll(".r-input", "input", (e) => {
                const idx = parseInt(e.target.dataset.idx);
                const val = parseFloat(e.target.value);
                let current = [...model.get("w_R")];
                current[idx] = val;
                model.set("w_R", current);
                e.target.previousElementSibling.innerText = val.toFixed(2);
                model.save_changes();
            });
        };

        updateUI();
        // Since we live-update values on input, we only want to re-render if features change
        // to avoid jumping/resetting the layout while sliding.
        // Actually, we do want to re-render if w_L/w_R change from python.
        // But for smooth dragging, we'll listen to change events, but let's be careful.
        
        let isDragging = false;
        el.addEventListener("mousedown", (e) => { if(e.target.type === 'range') isDragging = true; });
        document.addEventListener("mouseup", () => { isDragging = false; });
        
        model.on("change:features", updateUI);
        model.on("change:w_L", () => { if (!isDragging) updateUI(); });
        model.on("change:w_R", () => { if (!isDragging) updateUI(); });
    }
    export default { render };
    """

    _css = """
    .eq-container {
        font-family: system-ui, -apple-system, sans-serif;
        background: var(--gray-9, #111827);
        border: 2px solid var(--gray-7, #374151);
        border-radius: 12px;
        padding: 20px;
        color: var(--gray-1, #f9fafb);
        width: 100%;
        max-width: 100%;
        overflow-x: auto;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.3), 0 10px 15px -3px rgba(0,0,0,0.2);
    }
    
    .eq-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 24px;
        border-bottom: 1px solid var(--gray-7, #374151);
        padding-bottom: 12px;
    }
    
    .eq-title {
        font-weight: 700;
        font-size: 16px;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--gray-3, #d1d5db);
    }
    
    .eq-legend {
        font-size: 12px;
        display: flex;
        gap: 16px;
    }
    
    .eq-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 4px;
    }
    .dot-l { background: var(--blue-5, #3b82f6); box-shadow: 0 0 5px var(--blue-5); }
    .dot-r { background: var(--red-5, #ef4444); box-shadow: 0 0 5px var(--red-5); }
    
    .eq-board {
        display: flex;
        gap: 16px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    
    .eq-channel {
        display: flex;
        flex-direction: column;
        align-items: center;
        background: rgba(31, 41, 55, 0.5);
        border-radius: 8px;
        padding: 12px 8px;
        min-width: 70px;
        border: 1px solid var(--gray-8, #1f2937);
    }
    
    .eq-sliders {
        display: flex;
        gap: 8px;
        height: 200px;
    }
    
    .eq-slider-col {
        display: flex;
        flex-direction: column;
        align-items: center;
        height: 100%;
    }
    
    .eq-val {
        font-family: monospace;
        font-size: 10px;
        margin-bottom: 8px;
        height: 14px;
        opacity: 0.8;
    }
    .val-l { color: var(--blue-4, #60a5fa); }
    .val-r { color: var(--red-4, #f87171); }
    
    /* Vertical Range Slider Hack/Standard */
    input[type=range].eq-vertical {
        writing-mode: vertical-lr;
        direction: rtl;
        appearance: slider-vertical;
        width: 16px;
        height: 150px;
        margin: 0;
        cursor: grab;
    }
    
    input[type=range].eq-vertical:active {
        cursor: grabbing;
    }
    
    input[type=range].l-input {
        accent-color: var(--blue-5, #3b82f6);
    }
    input[type=range].r-input {
        accent-color: var(--red-5, #ef4444);
    }
    
    .eq-label {
        margin-top: 16px;
        font-size: 11px;
        width: 60px;
        text-align: center;
        color: var(--gray-4, #9ca3af);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        font-weight: 500;
        text-transform: uppercase;
        border-top: 1px solid var(--gray-7, #374151);
        padding-top: 8px;
    }

    /* Light Mode Support */
    @media (prefers-color-scheme: light) {
        .eq-container {
            background: var(--gray-1, #ffffff);
            border-color: var(--gray-3, #e5e7eb);
            color: var(--gray-8, #1f2937);
        }
        .eq-channel {
            background: var(--gray-2, #f9fafb);
            border-color: var(--gray-3, #e5e7eb);
        }
        .eq-header {
            border-bottom-color: var(--gray-3, #e5e7eb);
        }
        .eq-title {
            color: var(--gray-7, #374151);
        }
        .eq-label {
            color: var(--gray-6, #4b5563);
            border-top-color: var(--gray-3, #e5e7eb);
        }
    }
    """

    features = traitlets.List(traitlets.Unicode()).tag(sync=True)
    w_L = traitlets.List(traitlets.Float()).tag(sync=True)
    w_R = traitlets.List(traitlets.Float()).tag(sync=True)
