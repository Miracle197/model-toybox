/* TrainKit dashboard. Plain JS, no build step. Data comes from /api and a per-run WebSocket. */
(() => {
  "use strict";

  // ------------------------------------------------------------------ i18n
  const I18N = {
    zh: {
      newRun: "新建训练", runs: "训练记录", noRuns: "还没有训练记录", home: "概览",
      state_starting: "启动中", state_preparing: "准备中", state_running: "训练中", state_paused: "已暂停",
      state_saving: "本轮后保存退出", state_saved: "已保存", state_finished: "已完成", state_failed: "失败",
      state_stopped: "已停止", state_interrupted: "已中断",
      pause: "暂停", cont: "继续", saveStop: "本轮结束后保存并退出", stop: "立即停止", resume: "继续训练",
      bundle: "下载模型包", copyPlay: "复制试玩命令", copied: "已复制",
      confirmStop: "立即停止？当前这一轮的进度会丢失，之后可以从上一轮的存档继续。",
      epoch: "轮次", batch: "本轮", eta: "预计剩余", speed: "速度", device: "设备", batchSize: "批大小",
      dataset: "数据集", recipe: "模型", weights: "初始权重", elapsed: "已用时间", images: "图片",
      trainLoss: "训练损失", valLoss: "验证损失", metricsChart: "精度指标", lrChart: "学习率",
      sysChart: "设备占用", gpuUtil: "GPU 利用率", vram: "显存", cpu: "CPU", ram: "内存",
      samples: "验证集预测样例", log: "日志", table: "表格", charts: "图表", noData: "等待第一轮结束…",
      best: "最佳", prev: "上一轮", error: "错误", sampleNote: "第 {e} 轮的模型在固定验证图上的预测",
      tier_nvidia: "NVIDIA 训练档", tier_mps: "Apple MPS 训练档", tier_cpu: "CPU 预览档",
      welcome: "选一个训练记录，或新建一次训练。", envReady: "YOLO 环境", envMissing: "环境还没装好：在终端运行",
      system: "这台电脑", startTraining: "开始训练", name: "名称（可选）", epochs: "轮数", imgsz: "图片尺寸",
      batchAuto: "批大小（-1 自动）", datasetHint: "内置数据集名、已登记名称、文件夹/zip/data.yaml 路径，或 roboflow:workspace/project/版本",
      trainOk: "这台电脑可以训练", trainWarn: "可以训练，但硬件偏弱", trainNo: "这台电脑不建议训练",
      starting: "正在启动…", plots: "训练结束后的图表", paused_hint: "已暂停：显存仍被占用。想释放显卡请用「本轮结束后保存并退出」。",
      saving_hint: "当前这一轮结束后会保存并退出，之后可以继续训练。", resumable: "可以从第 {e} 轮的存档继续训练。",
      interrupted_hint: "训练进程意外结束（崩溃、关机或被结束）。", lastLines: "最近日志", recent: "最近的训练",
      openRun: "打开", val: "验证", train: "训练", finalEval: "最佳模型最终评估",
    },
    en: {
      newRun: "New training", runs: "Runs", noRuns: "No runs yet", home: "Overview",
      state_starting: "Starting", state_preparing: "Preparing", state_running: "Training", state_paused: "Paused",
      state_saving: "Saving after epoch", state_saved: "Saved", state_finished: "Finished", state_failed: "Failed",
      state_stopped: "Stopped", state_interrupted: "Interrupted",
      pause: "Pause", cont: "Continue", saveStop: "Save & exit after this epoch", stop: "Stop now", resume: "Continue training",
      bundle: "Download model bundle", copyPlay: "Copy try-it command", copied: "Copied",
      confirmStop: "Stop now? Progress in the current epoch is lost; you can continue from the last saved epoch.",
      epoch: "Epoch", batch: "Batch", eta: "ETA", speed: "Speed", device: "Device", batchSize: "Batch size",
      dataset: "Dataset", recipe: "Model", weights: "Start weights", elapsed: "Elapsed", images: "images",
      trainLoss: "Training loss", valLoss: "Validation loss", metricsChart: "Accuracy metrics", lrChart: "Learning rate",
      sysChart: "Device load", gpuUtil: "GPU util", vram: "VRAM", cpu: "CPU", ram: "RAM",
      samples: "Validation predictions", log: "Log", table: "Table", charts: "Charts", noData: "Waiting for the first epoch…",
      best: "best", prev: "prev", error: "Error", sampleNote: "Predictions of the epoch {e} model on fixed validation images",
      tier_nvidia: "NVIDIA training tier", tier_mps: "Apple MPS training tier", tier_cpu: "CPU preview tier",
      welcome: "Pick a run, or start a new training.", envReady: "YOLO environment", envMissing: "Environment not installed. In a terminal run",
      system: "This computer", startTraining: "Start training", name: "Name (optional)", epochs: "Epochs", imgsz: "Image size",
      batchAuto: "Batch (-1 = auto)", datasetHint: "Built-in name, registered name, folder/zip/data.yaml path, or roboflow:workspace/project/version",
      trainOk: "This computer can train", trainWarn: "Can train, but hardware is tight", trainNo: "Training not recommended here",
      starting: "Starting…", plots: "Plots after training", paused_hint: "Paused: GPU memory is still in use. To free the GPU use 'Save & exit after this epoch'.",
      saving_hint: "Will save and exit when this epoch ends; you can continue later.", resumable: "Can continue from the epoch {e} checkpoint.",
      interrupted_hint: "The training process ended unexpectedly (crash, shutdown or killed).", lastLines: "Latest log", recent: "Recent runs",
      openRun: "Open", val: "val", train: "train", finalEval: "Final evaluation of the best model",
    },
  };
  let lang = (() => { try { return localStorage.getItem("tk_lang"); } catch { return null; } })()
    || ((navigator.language || "").toLowerCase().startsWith("zh") ? "zh" : "en");
  const loc = (v, fallback = "") => (v && typeof v === "object" ? v[lang] || v.en || fallback : v || fallback);
  const t = (k, vars) => {
    let s = (I18N[lang] && I18N[lang][k]) || I18N.en[k] || k;
    if (vars) for (const [a, b] of Object.entries(vars)) s = s.replace(`{${a}}`, b);
    return s;
  };

  // ------------------------------------------------------------------ helpers
  const $ = (sel, el = document) => el.querySelector(sel);
  const h = (tag, attrs = {}, ...kids) => {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v == null || v === false) continue;
      if (k === "class") el.className = v;
      else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else if (k === "html") el.innerHTML = v;
      else el.setAttribute(k, v === true ? "" : v);
    }
    for (const kid of kids.flat()) if (kid != null && kid !== false) el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
    return el;
  };
  const api = async (path, opts = {}) => {
    const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
    if (!res.ok) {
      let msg = res.statusText;
      try { msg = (await res.json()).detail || msg; } catch { /* plain text */ }
      throw new Error(msg);
    }
    return res.json();
  };
  const fmtSecs = (s) => {
    if (s == null || !isFinite(s)) return "—";
    s = Math.round(s);
    const hh = Math.floor(s / 3600), mm = Math.floor((s % 3600) / 60), ss = s % 60;
    return hh ? `${hh}:${String(mm).padStart(2, "0")}:${String(ss).padStart(2, "0")}` : `${mm}:${String(ss).padStart(2, "0")}`;
  };
  const fmt = (v, d = 3) => (v == null || !isFinite(v) ? "—" : Number(v).toFixed(d));
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const toast = (msg) => {
    const el = $("#toast");
    el.textContent = msg; el.hidden = false;
    clearTimeout(toast._t); toast._t = setTimeout(() => (el.hidden = true), 3200);
  };
  const STATE_KIND = { running: "active", preparing: "active", starting: "active", paused: "warning", saving: "warning",
    saved: "good", finished: "good", failed: "critical", stopped: "serious", interrupted: "serious" };
  const badge = (state) => h("span", { class: `badge ${STATE_KIND[state] || ""}` }, h("span", { class: "dot", "aria-hidden": "true" }), t(`state_${state}`));
  const ACTIVE = new Set(["starting", "preparing", "running", "paused", "saving"]);

  // ------------------------------------------------------------------ global state
  const S = { system: null, recipes: [], datasets: null, runs: [], route: "", ws: null, charts: [], view: null,
    sysHist: [], timers: [] };

  // ------------------------------------------------------------------ charts (uPlot + hover tooltip)
  // Color follows the entity: a loss component keeps its color in every chart.
  const LOSS_SLOT = { box: 1, cls: 2, dfl: 3, seg: 4, pose: 4, kobj: 5, angle: 4, loss: 1, rle: 5 };
  const lossSlot = (key) => LOSS_SLOT[key.split("/").pop().replace("_loss", "")] || 5;

  function tooltipPlugin(fmtX) {
    let tip;
    return {
      hooks: {
        init: (u) => { tip = h("div", { class: "tip", hidden: true }); u.over.appendChild(tip); },
        setCursor: (u) => {
          const i = u.cursor.idx;
          if (i == null || u.cursor.left < 0) { tip.hidden = true; return; }
          tip.replaceChildren(h("div", { class: "x" }, fmtX(u.data[0][i])));
          u.series.forEach((s, si) => {
            if (si === 0 || !s.show) return;
            const v = u.data[si][i];
            tip.append(h("div", { class: "row" }, h("span", { class: "key", style: `background:${s._stroke}` }),
              `${s.label}: `, h("b", {}, v == null ? "—" : fmt(v, Math.abs(v) < 0.01 ? 5 : 4))));
          });
          tip.hidden = false;
          const w = tip.offsetWidth, left = u.cursor.left + 14 + w > u.over.clientWidth ? u.cursor.left - w - 14 : u.cursor.left + 14;
          tip.style.left = `${left}px`; tip.style.top = `${Math.max(0, u.cursor.top - 10)}px`;
        },
      },
    };
  }

  function makeChart(el, series, fmtX = (x) => `${t("epoch")} ${x}`, opts = {}) {
    const muted = css("--text-muted"), grid = css("--grid"), axis = css("--axis");
    const axisCfg = { stroke: muted, grid: { stroke: grid, width: 1 }, ticks: { stroke: axis, width: 1, size: 4 }, font: "11px system-ui" };
    const cfg = {
      width: Math.max(el.clientWidth, 200), height: 200,
      scales: { x: { time: false }, y: opts.y || {} },
      legend: { show: series.length > 1, live: false },
      cursor: { drag: { x: false, y: false }, points: { size: 8 } },
      axes: [{ ...axisCfg, ...(opts.xAxis || {}) }, { ...axisCfg, size: 52, ...(opts.yAxis || {}) }],
      series: [{ label: "x" }, ...series.map((s) => {
        const stroke = css(`--series-${s.slot}`);
        return { label: s.label, stroke, _stroke: stroke, width: 2, points: { show: opts.points !== false, size: 5, fill: stroke }, spanGaps: true };
      })],
      plugins: [tooltipPlugin(fmtX)],
    };
    const u = new uPlot(cfg, [[], ...series.map(() => [])], el);
    const ro = new ResizeObserver(() => u.setSize({ width: Math.max(el.clientWidth, 200), height: 200 }));
    ro.observe(el);
    u._ro = ro;
    S.charts.push(u);
    return u;
  }

  function destroyCharts() {
    for (const c of S.charts) { try { c._ro.disconnect(); c.destroy(); } catch { /* already gone */ } }
    S.charts = [];
  }

  // ------------------------------------------------------------------ sidebar + topbar
  async function refreshRuns() {
    try { S.runs = await api("/api/runs"); } catch { return; }
    const list = $("#runlist");
    if (!S.runs.length) { list.replaceChildren(h("div", { class: "empty" }, t("noRuns"))); return; }
    const current = S.route.startsWith("#/run/") ? decodeURIComponent(S.route.slice(6)) : "";
    list.replaceChildren(...S.runs.map((r) => {
      const pct = r.epochs ? Math.min(100, (100 * r.epoch) / r.epochs) : 0;
      return h("a", { class: `runitem${r.id === current ? " active" : ""}`, href: `#/run/${encodeURIComponent(r.id)}` },
        h("div", { class: "name", title: r.id }, r.name || r.id),
        h("div", { class: "sub" }, h("span", {}, `${r.recipe} · ${r.data_label || ""}`), badge(r.state)),
        h("div", { class: "mini" }, h("i", { style: `width:${pct}%` })));
    }));
  }

  async function loadSystem() {
    try { S.system = await api("/api/system"); } catch { return; }
    const sys = S.system.system;
    const gpu = (sys.gpus || [])[0];
    $("#device").textContent = `${t("tier_" + sys.tier)}${gpu ? " · " + gpu.name : ""}`;
  }

  function renderSysMini(g) {
    const el = $("#sysmini");
    const parts = [];
    if (g.gpus && g.gpus.length) {
      const x = g.gpus[0];
      parts.push(h("span", {}, "GPU ", h("b", {}, `${x.util}%`)), h("span", {}, `${t("vram")} `, h("b", {}, `${x.mem_used_gb}/${x.mem_total_gb} GB`)));
      if (x.temp != null) parts.push(h("span", {}, h("b", {}, `${x.temp}°C`)));
    }
    parts.push(h("span", {}, "CPU ", h("b", {}, `${Math.round(g.cpu)}%`)), h("span", {}, `${t("ram")} `, h("b", {}, `${Math.round(g.ram)}%`)));
    el.replaceChildren(...parts);
  }

  // ------------------------------------------------------------------ views
  function clearView() {
    if (S.ws) { S.ws.onclose = null; S.ws.close(); S.ws = null; }
    for (const id of S.timers) clearInterval(id);
    S.timers = [];
    destroyCharts();
    S.view = null;
  }

  async function showHome() {
    const main = $("#main");
    if (!S.system) await loadSystem();
    const sys = S.system?.system || {};
    const env = S.system?.envs?.ultralytics;
    const gpu = (sys.gpus || [])[0];
    const kv = h("dl", { class: "kv" },
      h("dt", {}, "OS"), h("dd", {}, `${sys.os || ""} ${sys.os_version || ""}`),
      h("dt", {}, "CPU"), h("dd", {}, `${sys.cpu || ""} · ${sys.cpu_count || ""}`),
      h("dt", {}, t("ram")), h("dd", {}, `${sys.ram_gb || "?"} GB`),
      h("dt", {}, "GPU"), h("dd", {}, gpu ? `${gpu.name}${gpu.vendor === "nvidia" ? ` · ${gpu.vram_gb} GB` : ""}` : "—"),
      h("dt", {}, t("envReady")), h("dd", {}, env && env.ok ? `${env.profile} · torch ${env.self_test?.torch || "?"} · ${env.self_test?.device || "?"}` : "—"));
    const envAlert = env && env.ok ? null : h("div", { class: "alert warning section" }, h("span", { class: "icon" }, "!"),
      h("div", {}, t("envMissing"), " ", h("code", {}, `trainkit setup yolo-detect --profile ${S.system?.profile || "train"}`)));
    const recent = S.runs.slice(0, 5);
    main.replaceChildren(
      h("div", { class: "runhead" }, h("div", {}, h("h1", {}, t("home")), h("div", { class: "meta" }, t("welcome"))),
        h("a", { class: "btn primary", href: "#/new" }, t("newRun"))),
      h("div", { class: "grid cols-2" },
        h("section", { class: "card" }, h("h3", {}, `${t("system")} · ${t("tier_" + (sys.tier || "cpu"))}`), kv),
        h("section", { class: "card" }, h("h3", {}, t("recent")), recent.length ? h("table", { class: "data" },
          h("tbody", {}, ...recent.map((r) => h("tr", {}, h("td", {}, h("a", { href: `#/run/${encodeURIComponent(r.id)}` }, r.name || r.id)),
            h("td", {}, badge(r.state)), h("td", {}, `${r.epoch}/${r.epochs || "?"}`))))) : h("div", { class: "empty" }, t("noRuns")))),
      envAlert || "");
  }

  async function showNew() {
    const main = $("#main");
    if (!S.recipes.length) S.recipes = await api("/api/recipes");
    if (!S.datasets) S.datasets = await api("/api/datasets");
    if (!S.system) await loadSystem();
    const recipeSel = h("select", { id: "f-recipe" }, ...S.recipes.map((r) => h("option", { value: r.id }, `${loc(r.i18n?.title, r.title)} (${r.id})`)));
    const weightsSel = h("select", { id: "f-weights" });
    const dsInput = h("input", { id: "f-data", list: "f-data-list", placeholder: "coco8.yaml", required: true, autocomplete: "off" });
    const dsList = h("datalist", { id: "f-data-list" });
    const epochs = h("input", { id: "f-epochs", type: "number", min: 1 });
    const imgsz = h("input", { id: "f-imgsz", type: "number", min: 32, step: 32 });
    const batch = h("input", { id: "f-batch", type: "number", step: "any", value: -1 });
    const name = h("input", { id: "f-name", maxlength: 60 });
    const verdict = h("div", { class: "hint" });
    const errBox = h("div");
    const sync = () => {
      const r = S.recipes.find((x) => x.id === recipeSel.value);
      weightsSel.replaceChildren(...r.weights.choices.map((w) => h("option", { value: w, selected: w === r.default_weights }, w)));
      const reg = Object.entries(S.datasets.registered || {}).filter(([, v]) => v.task === r.task).map(([k]) => h("option", { value: k }, k));
      dsList.replaceChildren(...reg, ...r.builtin_datasets.map((d) => h("option", { value: d.name }, loc(d.i18n_note, d.note))));
      dsInput.placeholder = r.builtin_datasets[0]?.name || "";
      epochs.value = r.defaults.epochs ?? 100;
      imgsz.value = r.defaults.imgsz ?? 640;
      batch.value = r.defaults.batch ?? -1;
      const v = S.system?.verdicts?.[r.id];
      if (v) verdict.textContent = `${v.train === "ok" ? "✓ " + t("trainOk") : v.train === "warn" ? "! " + t("trainWarn") : "✗ " + t("trainNo")}${v.reasons.length ? " — " + v.reasons.join("; ") : ""}`;
    };
    recipeSel.addEventListener("change", sync);
    const submit = h("button", { class: "primary", type: "submit" }, t("startTraining"));
    const form = h("form", { class: "card", onsubmit: async (e) => {
      e.preventDefault();
      submit.disabled = true; submit.textContent = t("starting"); errBox.replaceChildren();
      try {
        const res = await api("/api/runs", { method: "POST", body: JSON.stringify({
          recipe: recipeSel.value, weights: weightsSel.value, data: dsInput.value || dsInput.placeholder,
          epochs: Number(epochs.value), imgsz: Number(imgsz.value), batch: Number(batch.value), name: name.value || null }) });
        await refreshRuns();
        location.hash = `#/run/${encodeURIComponent(res.id)}`;
      } catch (err) {
        errBox.replaceChildren(h("div", { class: "alert critical" }, h("span", { class: "icon" }, "✗"), h("div", {}, err.message)));
        submit.disabled = false; submit.textContent = t("startTraining");
      }
    } },
      h("div", { class: "field" }, h("label", { for: "f-recipe" }, t("recipe")), recipeSel, verdict),
      h("div", { class: "field" }, h("label", { for: "f-weights" }, t("weights")), weightsSel),
      h("div", { class: "field" }, h("label", { for: "f-data" }, t("dataset")), dsInput, dsList, h("div", { class: "hint" }, t("datasetHint"))),
      h("div", { class: "row3" },
        h("div", { class: "field" }, h("label", { for: "f-epochs" }, t("epochs")), epochs),
        h("div", { class: "field" }, h("label", { for: "f-imgsz" }, t("imgsz")), imgsz),
        h("div", { class: "field" }, h("label", { for: "f-batch" }, t("batchAuto")), batch)),
      h("div", { class: "field" }, h("label", { for: "f-name" }, t("name")), name),
      errBox, h("div", { class: "actions section" }, submit));
    main.replaceChildren(h("div", { class: "runhead" }, h("h1", {}, t("newRun"))), form);
    sync();
  }

  // ------------------------------------------------------------------ run view
  function metricKeys(task) {
    if (task === "classify") return [["metrics/accuracy_top1", "top1", 1], ["metrics/accuracy_top5", "top5", 2]];
    const sfx = task === "segment" ? "(M)" : task === "pose" ? "(P)" : "(B)";
    return [[`metrics/mAP50${sfx}`, "mAP50", 1], [`metrics/mAP50-95${sfx}`, "mAP50-95", 2],
      [`metrics/precision${sfx}`, "precision", 3], [`metrics/recall${sfx}`, "recall", 4]];
  }

  async function showRun(id) {
    const main = $("#main");
    let payload;
    try { payload = await api(`/api/runs/${encodeURIComponent(id)}`); } catch (e) {
      main.replaceChildren(h("div", { class: "alert critical" }, h("span", { class: "icon" }, "✗"), h("div", {}, e.message)));
      return;
    }
    const meta = payload.meta;
    const V = (S.view = { id, meta, summary: payload.summary, resumable: payload.resumable, hasWeights: payload.has_weights,
      plots: payload.plots, tableMode: false, chartsBuilt: false, seriesKey: "" });

    V.el = {
      badge: h("span"), facts: h("div", { class: "facts" }), hint: h("div"),
      epochBar: h("i"), epochNum: h("span", { class: "num" }), batchBar: h("i"), batchNum: h("span", { class: "num" }),
      tiles: h("div", { class: "grid cols-4 section" }), charts: h("div", { class: "grid cols-2 section" }),
      table: h("div", { class: "card section table-wrap", hidden: true }), samples: h("section", { class: "card section", hidden: true }),
      plots: h("section", { class: "card section", hidden: true }), log: h("pre", { class: "log mono" }), actions: h("div", { class: "actions" }),
    };
    const toggle = h("button", { class: "small ghost", type: "button", onclick: () => {
      V.tableMode = !V.tableMode; toggle.textContent = V.tableMode ? t("charts") : t("table");
      V.el.table.hidden = !V.tableMode; V.el.charts.hidden = V.tableMode; renderTable();
    } }, t("table"));

    main.replaceChildren(
      h("div", { class: "runhead" },
        h("div", {}, h("div", { style: "display:flex;gap:10px;align-items:center;flex-wrap:wrap" }, h("h1", {}, meta.name || id), V.el.badge),
          h("div", { class: "meta" }, `${meta.recipe} · ${meta.weights} · ${t("dataset")}: ${meta.data_label}`)),
        V.el.actions),
      V.el.hint,
      h("section", { class: "card" },
        h("div", { class: "progress" }, h("span", {}, t("epoch")), h("div", { class: "bar" }, V.el.epochBar), V.el.epochNum),
        h("div", { class: "progress" }, h("span", {}, t("batch")), h("div", { class: "bar" }, V.el.batchBar), V.el.batchNum),
        V.el.facts),
      V.el.tiles,
      h("div", { class: "chart-head section" }, h("h3", { style: "margin:0" }, t("charts")), toggle),
      V.el.charts, V.el.table, V.el.samples, V.el.plots,
      h("section", { class: "card section" }, h("h3", {}, t("lastLines")), V.el.log));

    updateRun();
    connectWs(id);
    pollLog();
    S.timers.push(setInterval(pollLog, 3000));
  }

  function buildCharts() {
    const V = S.view, s = V.summary;
    const hist = s.history || [];
    const trainKeys = [...new Set(hist.flatMap((e) => Object.keys(e.losses || {})))];
    const valKeys = [...new Set(hist.flatMap((e) => Object.keys(e.metrics || {}).filter((k) => k.startsWith("val/"))))];
    const mKeys = metricKeys(s.task || V.meta.task).filter(([k]) => hist.some((e) => k in (e.metrics || {})));
    const key = [trainKeys, valKeys, mKeys.map((m) => m[0])].join("|");
    if (V.chartsBuilt && key === V.seriesKey) return;
    destroyCharts();
    V.seriesKey = key; V.chartsBuilt = true;
    const card = (title) => { const body = h("div", { class: "chart" }); return [h("section", { class: "card chart-card" }, h("h3", {}, title), body), body]; };
    const cards = [];
    V.charts = {};
    if (trainKeys.length) {
      const [c, b] = card(t("trainLoss")); cards.push(c);
      V.charts.train = { keys: trainKeys, body: b, series: trainKeys.map((k) => ({ label: k.replace("train/", ""), slot: lossSlot(k) })) };
    }
    if (valKeys.length) {
      const [c, b] = card(t("valLoss")); cards.push(c);
      V.charts.val = { keys: valKeys, body: b, series: valKeys.map((k) => ({ label: k.replace("val/", ""), slot: lossSlot(k) })) };
    }
    if (mKeys.length) {
      const [c, b] = card(t("metricsChart")); cards.push(c);
      V.charts.metrics = { keys: mKeys.map((m) => m[0]), body: b, series: mKeys.map((m) => ({ label: m[1], slot: m[2] })), y: { range: [0, 1] } };
    }
    const [lc, lb] = card(t("lrChart")); cards.push(lc);
    V.charts.lr = { keys: ["lr/pg0"], body: lb, series: [{ label: "lr", slot: 1 }],
      yAxis: { values: (u, v) => v.map((x) => (x === 0 ? "0" : x.toExponential(1))) } };
    const [sc, sb] = card(t("sysChart")); cards.push(sc);
    V.el.charts.replaceChildren(...cards);
    if (!hist.length) {
      for (const c of Object.values(V.charts)) c.body.replaceChildren(h("div", { class: "nodata" }, t("noData")));
    } else {
      for (const c of Object.values(V.charts)) c.u = makeChart(c.body, c.series, undefined, { y: c.y, yAxis: c.yAxis });
    }
    const hasGpu = (S.lastGpu?.gpus || []).length > 0;
    V.sysChart = makeChart(sb, hasGpu ? [{ label: t("gpuUtil"), slot: 1 }, { label: `${t("vram")} %`, slot: 2 }]
      : [{ label: `${t("cpu")} %`, slot: 1 }, { label: `${t("ram")} %`, slot: 2 }],
    (x) => `${Math.round(x)}s`, { y: { range: [0, 100] }, points: false, xAxis: { values: (u, v) => v.map((x) => `${Math.round(x)}s`) } });
    V.sysHasGpu = hasGpu;
  }

  function updateCharts() {
    const V = S.view, hist = V.summary.history || [];
    buildCharts();
    if (!hist.length) return;
    const x = hist.map((e) => e.epoch);
    for (const [name, c] of Object.entries(V.charts)) {
      if (!c.u) continue;
      const src = (e) => (name === "train" ? e.losses : name === "lr" ? e.lr : e.metrics) || {};
      c.u.setData([x, ...c.keys.map((k) => hist.map((e) => (k in src(e) ? src(e)[k] : null)))]);
    }
  }

  function updateSysChart() {
    const V = S.view;
    if (!V || !V.sysChart || !S.sysHist.length) return;
    const now = S.sysHist[S.sysHist.length - 1].t;
    const xs = S.sysHist.map((g) => g.t - now);
    const a = S.sysHist.map((g) => (V.sysHasGpu ? g.gpus?.[0]?.util ?? null : g.cpu));
    const b = S.sysHist.map((g) => (V.sysHasGpu ? g.gpus?.[0]?.mem ?? null : g.ram));
    V.sysChart.setData([xs, a, b]);
  }

  function renderTable() {
    const V = S.view;
    if (!V || !V.tableMode) return;
    const hist = V.summary.history || [];
    const mk = metricKeys(V.summary.task || V.meta.task);
    const lossKeys = [...new Set(hist.flatMap((e) => Object.keys(e.losses || {})))];
    V.el.table.replaceChildren(h("table", { class: "data" },
      h("thead", {}, h("tr", {}, h("th", {}, t("epoch")), ...lossKeys.map((k) => h("th", {}, k.replace("train/", ""))),
        ...mk.map((m) => h("th", {}, m[1])), h("th", {}, "s"))),
      h("tbody", {}, ...hist.slice().reverse().map((e) => h("tr", {}, h("td", {}, e.epoch),
        ...lossKeys.map((k) => h("td", {}, fmt(e.losses?.[k], 4))),
        ...mk.map((m) => h("td", {}, fmt(e.metrics?.[m[0]], 4))), h("td", {}, fmt(e.seconds, 1)))))));
  }

  function renderTiles() {
    const V = S.view, s = V.summary, hist = s.history || [];
    const mk = metricKeys(s.task || V.meta.task);
    const final = s.final && Object.keys(s.final).length ? s.final : null;
    const last = final || hist[hist.length - 1]?.metrics || {};
    const prev = hist.length > 1 ? hist[hist.length - 2].metrics : null;
    const tiles = mk.filter(([k]) => k in last).map(([k, label]) => {
      const v = last[k];
      const best = Math.max(...hist.map((e) => e.metrics?.[k] ?? -1));
      const d = prev && !final ? v - (prev[k] ?? v) : null;
      return h("section", { class: "card tile" }, h("div", { class: "label" }, label + (final ? ` · ${t("finalEval")}` : "")),
        h("div", { class: "value" }, fmt(v, 3)),
        h("div", { class: `delta${d > 0 ? " up" : ""}` }, d != null ? `${Math.abs(d) < 5e-4 ? "=" : d > 0 ? "▲" : "▼"} ${fmt(Math.abs(d), 3)} ${t("prev")} · ${t("best")} ${fmt(best, 3)}` : `${t("best")} ${fmt(best, 3)}`));
    });
    V.el.tiles.replaceChildren(...tiles);
    V.el.tiles.hidden = !tiles.length;
  }

  function renderActions() {
    const V = S.view, s = V.summary, st = s.state;
    const ctl = (cmd, label, cls = "") => h("button", { class: cls, type: "button", onclick: async (e) => {
      if (cmd === "stop" && !confirm(t("confirmStop"))) return;
      e.currentTarget.disabled = true;
      try { await api(`/api/runs/${encodeURIComponent(V.id)}/control`, { method: "POST", body: JSON.stringify({ cmd }) }); }
      catch (err) { toast(err.message); }
    } }, label);
    const btns = [];
    if (st === "running" || st === "preparing" || st === "starting") btns.push(ctl("pause", t("pause")));
    if (st === "paused") btns.push(ctl("continue", t("cont"), "primary"));
    if (st === "running" || st === "paused") btns.push(ctl("save_stop", t("saveStop")));
    if (ACTIVE.has(st)) btns.push(ctl("stop", t("stop"), "danger"));
    if (V.resumable) btns.push(h("button", { class: "primary", type: "button", onclick: async (e) => {
      e.currentTarget.disabled = true;
      try { await api(`/api/runs/${encodeURIComponent(V.id)}/resume`, { method: "POST", body: "{}" }); toast(t("starting")); }
      catch (err) { toast(err.message); e.currentTarget.disabled = false; }
    } }, t("resume")));
    if (V.hasWeights && !ACTIVE.has(st)) {
      btns.push(h("a", { class: "btn", href: `/api/runs/${encodeURIComponent(V.id)}/bundle` }, t("bundle")));
      btns.push(h("button", { type: "button", class: "ghost", onclick: async () => {
        const cmd = `trainkit play ${V.id}`;
        try { await navigator.clipboard.writeText(cmd); toast(`${t("copied")}: ${cmd}`); } catch { toast(cmd); }
      } }, t("copyPlay")));
    }
    const sig = btns.map((b) => b.textContent).join("|");
    if (sig !== V.actionsSig) { V.el.actions.replaceChildren(...btns); V.actionsSig = sig; }
  }

  function updateRun() {
    const V = S.view;
    if (!V) return;
    const s = V.summary;
    V.el.badge.replaceChildren(badge(s.state));
    const inEpoch = s.current_epoch > s.epoch;
    const ePct = s.epochs ? (100 * (s.epoch + (inEpoch && s.nb ? s.batch / s.nb : 0))) / s.epochs : 0;
    V.el.epochBar.style.width = `${Math.min(100, ePct)}%`;
    V.el.epochNum.textContent = `${s.epoch} / ${s.epochs || "?"}`;
    V.el.batchBar.style.width = `${inEpoch && s.nb ? (100 * s.batch) / s.nb : 0}%`;
    V.el.batchNum.textContent = inEpoch ? `${s.batch} / ${s.nb || "?"}` : "—";
    const elapsed = s.started && s.updated ? s.updated - s.started : null;
    const losses = Object.entries(s.losses || {}).slice(0, 3).map(([k, v]) => h("span", {}, `${k.replace("train/", "")} `, h("b", {}, fmt(v, 4))));
    V.el.facts.replaceChildren(
      h("span", {}, `${t("eta")} `, h("b", {}, ACTIVE.has(s.state) ? fmtSecs(s.eta_seconds) : "—")),
      h("span", {}, `${t("speed")} `, h("b", {}, s.rate ? `${fmt(s.rate, 2)} it/s` : "—")),
      h("span", {}, `${t("device")} `, h("b", {}, s.device_name || s.device || "—")),
      h("span", {}, `${t("batchSize")} `, h("b", {}, s.batch_size ?? "—")),
      h("span", {}, `${t("elapsed")} `, h("b", {}, fmtSecs(elapsed))), ...losses);
    let hint = null;
    if (s.state === "failed" && s.error) hint = ["critical", "✗", s.error];
    else if (s.state === "paused") hint = ["warning", "!", t("paused_hint")];
    else if (s.state === "saving") hint = ["info", "i", t("saving_hint")];
    else if (s.state === "interrupted") hint = ["warning", "!", t("interrupted_hint") + (V.resumable ? " " + t("resumable", { e: s.checkpoint_epoch }) : "")];
    else if (V.resumable) hint = ["info", "i", t("resumable", { e: s.checkpoint_epoch })];
    V.el.hint.replaceChildren(...(hint ? [h("div", { class: `alert ${hint[0]}`, style: "margin-bottom:16px" }, h("span", { class: "icon" }, hint[1]), h("div", {}, hint[2]))] : []));
    renderActions();
    renderTiles();
    updateCharts();
    renderTable();
    // Samples
    if (s.samples && s.samples.length) {
      const ep = (s.samples[0].match(/epoch(\d+)/) || [])[1];
      V.el.samples.hidden = false;
      const sig = s.samples.join("|");
      if (sig !== V.samplesSig) {
        V.samplesSig = sig;
        V.el.samples.replaceChildren(h("h3", {}, `${t("samples")} · ${t("sampleNote", { e: Number(ep) })}`),
          h("div", { class: "samples" }, ...s.samples.map((p) => h("img", { src: `/api/runs/${encodeURIComponent(V.id)}/files/${p}`, alt: p, loading: "lazy" }))));
      }
    }
    if (V.plots && V.plots.length && !ACTIVE.has(s.state)) {
      V.el.plots.hidden = false;
      if (!V.plotsShown) {
        V.plotsShown = true;
        V.el.plots.replaceChildren(h("h3", {}, t("plots")), h("div", { class: "samples" },
          ...V.plots.filter((p) => /results|confusion_matrix_normalized|PR_curve|BoxPR/.test(p)).slice(0, 4)
            .map((p) => h("img", { src: `/api/runs/${encodeURIComponent(V.id)}/files/train/${p}`, alt: p, loading: "lazy" }))));
      }
    }
  }

  async function pollLog() {
    const V = S.view;
    if (!V) return;
    try {
      const { lines } = await api(`/api/runs/${encodeURIComponent(V.id)}/log?tail=40`);
      const atBottom = V.el.log.scrollTop + V.el.log.clientHeight >= V.el.log.scrollHeight - 8;
      V.el.log.textContent = lines.join("\n");
      if (atBottom) V.el.log.scrollTop = V.el.log.scrollHeight;
    } catch { /* ignore */ }
  }

  function connectWs(id) {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/api/runs/${encodeURIComponent(id)}/ws${location.search}`);
    S.ws = ws;
    ws.onmessage = (msg) => {
      const data = JSON.parse(msg.data);
      if (!S.view || S.view.id !== id) return;
      if (data.type === "summary") {
        const prevState = S.view.summary.state;
        S.view.summary = data.summary;
        S.view.resumable = data.resumable;
        S.view.hasWeights = data.has_weights;
        if (prevState !== data.summary.state) {
          refreshRuns();
          if (!ACTIVE.has(data.summary.state)) api(`/api/runs/${encodeURIComponent(id)}`).then((p) => { if (S.view) { S.view.plots = p.plots; updateRun(); } });
        }
        updateRun();
      } else if (data.type === "gpu") {
        onGpu(data.gpu);
      }
    };
    ws.onclose = () => { if (S.view && S.view.id === id) setTimeout(() => S.view && S.view.id === id && connectWs(id), 2000); };
  }

  function onGpu(g) {
    S.lastGpu = g;
    S.sysHist.push(g);
    if (S.sysHist.length > 300) S.sysHist.shift();
    renderSysMini(g);
    updateSysChart();
  }

  // ------------------------------------------------------------------ router
  async function route() {
    S.route = location.hash || "#/";
    clearView();
    await refreshRuns();
    if (S.route.startsWith("#/run/")) await showRun(decodeURIComponent(S.route.slice(6)));
    else if (S.route === "#/new") await showNew();
    else await showHome();
  }

  function applyLang() {
    document.documentElement.lang = lang === "zh" ? "zh" : "en";
    $("#lang").textContent = lang === "zh" ? "EN" : "中文";
    document.querySelectorAll("[data-i18n]").forEach((el) => (el.textContent = t(el.dataset.i18n)));
  }

  $("#lang").addEventListener("click", () => {
    lang = lang === "zh" ? "en" : "zh";
    try { localStorage.setItem("tk_lang", lang); } catch { /* private mode */ }
    applyLang(); loadSystem(); route();
  });
  window.addEventListener("hashchange", route);
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { if (S.view) { S.view.chartsBuilt = false; updateRun(); } });

  applyLang();
  loadSystem();
  route();
  setInterval(refreshRuns, 4000);
  // Keep the top-bar device load fresh when no run view is streaming it
  setInterval(async () => { if (!S.ws) { try { onGpu(await api("/api/gpu")); } catch { /* offline */ } } }, 2000);
})();
