const WORLD = 10;
const DEFAULT_START = { x: 1.0, y: 1.0 };
const DEFAULT_GOAL = { x: 8.8, y: 8.7 };

const canvas = document.getElementById("mapCanvas");
const ctx = canvas.getContext("2d");
const obstacleCount = document.getElementById("obstacleCount");
const pointerReadout = document.getElementById("pointerReadout");
const saveState = document.getElementById("saveState");
const brushRadius = document.getElementById("brushRadius");
const brushRadiusValue = document.getElementById("brushRadiusValue");
const controls = document.getElementById("controls");
const runStatus = document.getElementById("runStatus");
const outputLinks = document.getElementById("outputLinks");
const traceTable = document.getElementById("traceTable");
const previewImage = document.getElementById("previewImage");
const savedMapSelect = document.getElementById("savedMapSelect");
const mapLibraryStatus = document.getElementById("mapLibraryStatus");
const progressWrap = document.getElementById("progressWrap");
const progressBar = document.getElementById("progressBar");
const progressText = document.getElementById("progressText");
const progressMessage = document.getElementById("progressMessage");
const liquidCell = document.getElementById("liquidCell");

let obstacles = [];
let startPoint = { ...DEFAULT_START };
let goalPoint = { ...DEFAULT_GOAL };
let selectedIndex = -1;
let mode = "add";
let dragging = false;
let draggingPoint = null;

const controlDefs = [
  { id: "noise", label: "Sensör gürültüsü", min: 0, max: 0.45, step: 0.01, value: 0.18 },
  { id: "dropout", label: "Sensör dropout", min: 0, max: 0.45, step: 0.01, value: 0.12 },
  { id: "sensorBias", label: "Sensör ölçek bias", min: 0.7, max: 1.3, step: 0.01, value: 0.96 },
  { id: "speed", label: "Robot hızı", min: 0.12, max: 0.55, step: 0.01, value: 0.32 },
  { id: "maxSteps", label: "Simülasyon uzunluğu / maksimum step", min: 30, max: 1200, step: 1, value: 220 },
  { id: "safeMargin", label: "Güvenlik marjı", min: 0.02, max: 0.85, step: 0.01, value: 0.35 },
  { id: "plannerClearance", label: "Planner clearance", min: 0, max: 0.35, step: 0.01, value: 0.04 },
  { id: "plannerLookahead", label: "Waypoint bakış mesafesi", min: 0.25, max: 2.2, step: 0.05, value: 0.85 },
  { id: "plannerWeight", label: "Planner ağırlığı", min: 0, max: 2.5, step: 0.05, value: 1.0 },
  { id: "policyWeight", label: "Politika ağırlığı", min: 0, max: 1.5, step: 0.05, value: 0.35 },
  { id: "ncpHidden", label: "NCP nöron sayısı", min: 12, max: 96, step: 1, value: 32 },
  { id: "ncpSparsity", label: "NCP seyrekliği", min: 0.1, max: 0.9, step: 0.05, value: 0.5 },
  { id: "ncpBaselineScale", label: "Fixed baseline etkisi", min: 0, max: 1.5, step: 0.05, value: 1.0 },
  { id: "ncpResidualScale", label: "NCP residual etkisi", min: 0, max: 1.5, step: 0.05, value: 0.35 },
  { id: "ncpLearningRate", label: "NCP online öğrenme hızı", min: 0, max: 0.05, step: 0.001, value: 0.006 },
  { id: "successRadius", label: "Hedef toleransı", min: 0.2, max: 0.85, step: 0.01, value: 0.45 },
  { id: "seed", label: "Seed", min: 1, max: 999999, step: 1, value: 770005 },
];

const presets = {
  balanced: {
    noise: 0.18,
    dropout: 0.12,
    sensorBias: 0.96,
    speed: 0.32,
    maxSteps: 220,
    safeMargin: 0.35,
    plannerClearance: 0.04,
    plannerLookahead: 0.85,
    plannerWeight: 1.0,
    policyWeight: 0.35,
    ncpHidden: 32,
    ncpSparsity: 0.5,
    ncpBaselineScale: 1.0,
    ncpResidualScale: 0.35,
    ncpLearningRate: 0.006,
    successRadius: 0.45,
  },
  mazeSolver: {
    noise: 0.16,
    dropout: 0.10,
    sensorBias: 0.98,
    speed: 0.30,
    maxSteps: 480,
    safeMargin: 0.36,
    plannerClearance: 0.05,
    plannerLookahead: 1.15,
    plannerWeight: 1.45,
    policyWeight: 0.22,
    ncpHidden: 40,
    ncpSparsity: 0.45,
    ncpBaselineScale: 1.0,
    ncpResidualScale: 0.30,
    ncpLearningRate: 0.005,
    successRadius: 0.45,
  },
  highNoise: {
    noise: 0.28,
    dropout: 0.26,
    sensorBias: 0.86,
    speed: 0.29,
    maxSteps: 420,
    safeMargin: 0.42,
    plannerClearance: 0.07,
    plannerLookahead: 1.0,
    plannerWeight: 1.35,
    policyWeight: 0.25,
    ncpHidden: 48,
    ncpSparsity: 0.5,
    ncpBaselineScale: 1.0,
    ncpResidualScale: 0.40,
    ncpLearningRate: 0.004,
    successRadius: 0.50,
  },
  safetyFirst: {
    noise: 0.14,
    dropout: 0.08,
    sensorBias: 1.0,
    speed: 0.26,
    maxSteps: 620,
    safeMargin: 0.52,
    plannerClearance: 0.10,
    plannerLookahead: 0.9,
    plannerWeight: 1.65,
    policyWeight: 0.18,
    ncpHidden: 40,
    ncpSparsity: 0.35,
    ncpBaselineScale: 1.0,
    ncpResidualScale: 0.25,
    ncpLearningRate: 0.003,
    successRadius: 0.50,
  },
  aggressive: {
    noise: 0.18,
    dropout: 0.12,
    sensorBias: 0.96,
    speed: 0.42,
    maxSteps: 180,
    safeMargin: 0.24,
    plannerClearance: 0.02,
    plannerLookahead: 0.65,
    plannerWeight: 0.75,
    policyWeight: 0.55,
    ncpHidden: 28,
    ncpSparsity: 0.60,
    ncpBaselineScale: 0.85,
    ncpResidualScale: 0.55,
    ncpLearningRate: 0.010,
    successRadius: 0.45,
  },
};

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function verticalWall(x, yMin, yMax, gapCenter, gapWidth, r = 0.3, spacing = 0.56) {
  const list = [];
  for (let y = yMin; y <= yMax + 1e-6; y += spacing) {
    if (Math.abs(y - gapCenter) > gapWidth / 2) list.push({ x, y, r });
  }
  return list;
}

function horizontalWall(y, xMin, xMax, gapCenter, gapWidth, r = 0.3, spacing = 0.56) {
  const list = [];
  for (let x = xMin; x <= xMax + 1e-6; x += spacing) {
    if (Math.abs(x - gapCenter) > gapWidth / 2) list.push({ x, y, r });
  }
  return list;
}

function templateObstacles(kind) {
  if (kind === "blank") return [];
  if (kind === "zigzag") {
    return [
      ...verticalWall(2.8, 0.9, 8.8, 2.6, 1.25),
      ...verticalWall(5.0, 1.1, 9.1, 6.6, 1.2),
      ...verticalWall(7.1, 1.0, 8.6, 4.2, 1.15),
      { x: 4.0, y: 3.8, r: 0.38 },
      { x: 6.0, y: 7.7, r: 0.35 },
    ];
  }
  if (kind === "dense") {
    return [
      ...horizontalWall(2.7, 0.9, 8.7, 7.4, 1.15),
      ...horizontalWall(4.8, 1.2, 9.0, 2.4, 1.1),
      ...horizontalWall(6.8, 0.9, 8.8, 7.5, 1.15),
      { x: 3.6, y: 3.8, r: 0.42 },
      { x: 5.5, y: 5.8, r: 0.42 },
      { x: 7.8, y: 4.1, r: 0.35 },
    ];
  }
  if (kind === "utrap") {
    return [
      { x: 4.4, y: 4.0, r: 0.5 },
      { x: 5.1, y: 4.0, r: 0.5 },
      { x: 5.8, y: 4.0, r: 0.5 },
      { x: 4.1, y: 4.7, r: 0.5 },
      { x: 4.1, y: 5.4, r: 0.5 },
      { x: 5.9, y: 4.7, r: 0.5 },
      { x: 5.9, y: 5.4, r: 0.5 },
      { x: 4.8, y: 6.0, r: 0.46 },
      { x: 6.9, y: 5.9, r: 0.48 },
      { x: 7.2, y: 7.1, r: 0.4 },
    ];
  }
  if (kind === "sensor") {
    return [
      ...verticalWall(3.3, 0.9, 8.8, 3.4, 1.0, 0.28),
      ...verticalWall(5.9, 1.0, 9.0, 6.0, 1.0, 0.28),
      { x: 4.2, y: 4.5, r: 0.55 },
      { x: 4.7, y: 5.3, r: 0.45 },
      { x: 6.7, y: 4.5, r: 0.45 },
      { x: 7.3, y: 6.3, r: 0.45 },
      { x: 2.2, y: 6.8, r: 0.35 },
    ];
  }
  if (kind === "labyrinth") {
    return [
      ...horizontalWall(2.35, 0.9, 9.0, 8.0, 1.65, 0.28, 0.52),
      ...horizontalWall(4.15, 0.9, 8.8, 2.0, 1.65, 0.28, 0.52),
      ...horizontalWall(5.95, 1.1, 9.0, 8.0, 1.65, 0.28, 0.52),
      ...horizontalWall(7.7, 0.9, 8.5, 2.2, 1.65, 0.28, 0.52),
      ...verticalWall(3.55, 2.75, 3.75, 10.0, 0.0, 0.24, 0.48),
      ...verticalWall(6.35, 4.55, 5.55, 10.0, 0.0, 0.24, 0.48),
      ...verticalWall(4.85, 6.35, 7.3, 10.0, 0.0, 0.24, 0.48),
      { x: 5.1, y: 2.95, r: 0.24 },
      { x: 3.2, y: 5.25, r: 0.24 },
      { x: 6.6, y: 7.05, r: 0.24 },
    ];
  }
  return [];
}

function worldToCanvas(point) {
  return {
    x: (point.x / WORLD) * canvas.width,
    y: canvas.height - (point.y / WORLD) * canvas.height,
  };
}

function canvasToWorld(event) {
  const rect = canvas.getBoundingClientRect();
  const x = ((event.clientX - rect.left) / rect.width) * WORLD;
  const y = WORLD - ((event.clientY - rect.top) / rect.height) * WORLD;
  return { x: clamp(x, 0, WORLD), y: clamp(y, 0, WORLD) };
}

function obstacleAt(point) {
  for (let i = obstacles.length - 1; i >= 0; i -= 1) {
    const obs = obstacles[i];
    const d = Math.hypot(obs.x - point.x, obs.y - point.y);
    if (d <= obs.r + 0.16) return i;
  }
  return -1;
}

function specialPointAt(point) {
  if (Math.hypot(startPoint.x - point.x, startPoint.y - point.y) <= 0.28) return "start";
  if (Math.hypot(goalPoint.x - point.x, goalPoint.y - point.y) <= 0.32) return "goal";
  return null;
}

function setSpecialPoint(kind, point) {
  const target = kind === "start" ? startPoint : goalPoint;
  target.x = clamp(point.x, 0.25, WORLD - 0.25);
  target.y = clamp(point.y, 0.25, WORLD - 0.25);
  syncPointInputs();
  markDirty(kind === "start" ? "Başlangıç değişti" : "Hedef değişti");
  draw();
}

function syncPointInputs() {
  document.getElementById("startX").value = startPoint.x.toFixed(2);
  document.getElementById("startY").value = startPoint.y.toFixed(2);
  document.getElementById("goalX").value = goalPoint.x.toFixed(2);
  document.getElementById("goalY").value = goalPoint.y.toFixed(2);
}

function readPointInputs() {
  const numericOr = (id, fallback) => {
    const value = Number(document.getElementById(id).value);
    return Number.isFinite(value) ? value : fallback;
  };
  startPoint = {
    x: clamp(numericOr("startX", startPoint.x), 0.25, WORLD - 0.25),
    y: clamp(numericOr("startY", startPoint.y), 0.25, WORLD - 0.25),
  };
  goalPoint = {
    x: clamp(numericOr("goalX", goalPoint.x), 0.25, WORLD - 0.25),
    y: clamp(numericOr("goalY", goalPoint.y), 0.25, WORLD - 0.25),
  };
  syncPointInputs();
  markDirty("Başlangıç/hedef değişti");
  draw();
}

function markDirty(text = "Düzenlendi") {
  saveState.textContent = text;
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#fbfcfb";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.strokeStyle = "#dde5e2";
  ctx.lineWidth = 1;
  for (let i = 0; i <= WORLD; i += 1) {
    const p = (i / WORLD) * canvas.width;
    ctx.beginPath();
    ctx.moveTo(p, 0);
    ctx.lineTo(p, canvas.height);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, p);
    ctx.lineTo(canvas.width, p);
    ctx.stroke();
  }

  const safeMargin = Number(document.getElementById("safeMargin")?.value ?? 0.35);
  obstacles.forEach((obs, idx) => {
    const p = worldToCanvas(obs);
    const r = (obs.r / WORLD) * canvas.width;
    const ring = ((obs.r + safeMargin) / WORLD) * canvas.width;
    ctx.beginPath();
    ctx.arc(p.x, p.y, ring, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(204, 75, 75, 0.11)";
    ctx.fill();
    ctx.beginPath();
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    ctx.fillStyle = idx === selectedIndex ? "#16837a" : "#9f2f3f";
    ctx.fill();
    ctx.lineWidth = idx === selectedIndex ? 3 : 1.5;
    ctx.strokeStyle = idx === selectedIndex ? "#0f625c" : "#6b222d";
    ctx.stroke();
  });

  const start = worldToCanvas(startPoint);
  ctx.beginPath();
  ctx.arc(start.x, start.y, 9, 0, Math.PI * 2);
  ctx.fillStyle = "#222222";
  ctx.fill();
  ctx.lineWidth = draggingPoint === "start" || mode === "start" ? 3 : 1.5;
  ctx.strokeStyle = "#ffffff";
  ctx.stroke();
  ctx.fillText("start", start.x + 10, start.y - 8);

  const goal = worldToCanvas(goalPoint);
  ctx.fillStyle = "#16837a";
  ctx.beginPath();
  for (let i = 0; i < 5; i += 1) {
    const angle = -Math.PI / 2 + (i * Math.PI * 2) / 5;
    const x = goal.x + Math.cos(angle) * 12;
    const y = goal.y + Math.sin(angle) * 12;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.fill();
  ctx.lineWidth = draggingPoint === "goal" || mode === "goal" ? 3 : 1.5;
  ctx.strokeStyle = "#0f625c";
  ctx.stroke();
  ctx.fillText("goal", goal.x + 12, goal.y - 10);

  obstacleCount.textContent = String(obstacles.length);
}

function updateBrushOutput() {
  brushRadiusValue.textContent = Number(brushRadius.value).toFixed(2);
  if (selectedIndex >= 0) {
    obstacles[selectedIndex].r = Number(brushRadius.value);
    markDirty();
    draw();
  }
}

function buildControls() {
  controls.innerHTML = "";
  controlDefs.forEach((def) => {
    const row = document.createElement("div");
    row.className = "control-row";
    row.innerHTML = `
      <div class="range-head">
        <label for="${def.id}">${def.label}</label>
        <output id="${def.id}Value">${def.value}</output>
      </div>
      <div class="control-editor">
        <input id="${def.id}" type="range" min="${def.min}" max="${def.max}" step="${def.step}" value="${def.value}" />
        <input id="${def.id}Number" class="control-number" type="number" min="${def.min}" max="${def.max}" step="${def.step}" value="${def.value}" />
      </div>
    `;
    controls.appendChild(row);
    const input = row.querySelector(`input[type="range"]`);
    const number = row.querySelector(`input[type="number"]`);
    const output = row.querySelector("output");
    const sync = (value) => {
      const parsed = clamp(Number(value), Number(def.min), Number(def.max));
      const next = Number.isFinite(parsed) ? parsed : def.value;
      input.value = next;
      number.value = next;
      output.textContent = next;
      markDirty("Parametre değişti");
      draw();
    };
    input.addEventListener("input", () => {
      sync(input.value);
    });
    number.addEventListener("change", () => {
      sync(number.value);
    });
  });
}

function setParams(values) {
  Object.entries(values).forEach(([key, value]) => {
    if (key === "liquidCell" && liquidCell) {
      liquidCell.value = value;
      return;
    }
    const input = document.getElementById(key);
    const number = document.getElementById(`${key}Number`);
    const output = document.getElementById(`${key}Value`);
    if (input) input.value = value;
    if (number) number.value = value;
    if (output) output.textContent = value;
  });
  draw();
}

function collectParams() {
  const params = {};
  controlDefs.forEach((def) => {
    const value = Number(document.getElementById(def.id).value);
    params[def.id] = ["seed", "maxSteps", "ncpHidden"].includes(def.id) ? Math.round(value) : value;
  });
  params.makeGif = document.getElementById("makeGif").checked;
  params.liquidCell = liquidCell.value;
  return params;
}

function selectedControllers() {
  const controllers = [];
  if (document.getElementById("ctrlFixed").checked) controllers.push("fixed");
  if (document.getElementById("ctrlLiquid").checked) controllers.push("liquid");
  if (document.getElementById("ctrlSupervisor").checked) controllers.push("supervisor");
  return controllers.length ? controllers : ["supervisor"];
}

function currentMapData() {
  return {
    name: document.getElementById("mapName").value,
    start: startPoint,
    goal: goalPoint,
    obstacles,
    params: collectParams(),
    controllers: selectedControllers(),
  };
}

function applyMapData(data) {
  document.getElementById("mapName").value = data.name || "custom_map";
  obstacles = Array.isArray(data.obstacles) ? data.obstacles : [];
  startPoint = data.start ? { x: Number(data.start.x), y: Number(data.start.y) } : { ...DEFAULT_START };
  goalPoint = data.goal ? { x: Number(data.goal.x), y: Number(data.goal.y) } : { ...DEFAULT_GOAL };
  selectedIndex = -1;
  dragging = false;
  draggingPoint = null;
  if (data.params) setParams(data.params);
  if (Array.isArray(data.controllers)) {
    document.getElementById("ctrlFixed").checked = data.controllers.includes("fixed");
    document.getElementById("ctrlLiquid").checked = data.controllers.includes("liquid");
    document.getElementById("ctrlSupervisor").checked = data.controllers.includes("supervisor");
  }
  syncPointInputs();
  draw();
}

function setMode(nextMode) {
  mode = nextMode;
  document.querySelectorAll(".mode").forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === mode);
  });
}

canvas.addEventListener("mousedown", (event) => {
  const point = canvasToWorld(event);
  const hit = obstacleAt(point);
  if (event.button === 2) {
    event.preventDefault();
    if (hit >= 0) {
      obstacles.splice(hit, 1);
      selectedIndex = -1;
      markDirty();
      draw();
    }
    return;
  }
  if (mode === "start" || mode === "goal") {
    selectedIndex = -1;
    draggingPoint = mode;
    setSpecialPoint(mode, point);
    return;
  }
  if (mode === "erase") {
    if (hit >= 0) {
      obstacles.splice(hit, 1);
      selectedIndex = -1;
      markDirty();
      draw();
    }
    return;
  }
  if (mode === "select") {
    draggingPoint = specialPointAt(point);
    if (draggingPoint) {
      selectedIndex = -1;
      setSpecialPoint(draggingPoint, point);
      return;
    }
    selectedIndex = hit;
    dragging = hit >= 0;
    if (hit >= 0) brushRadius.value = obstacles[hit].r;
    updateBrushOutput();
    draw();
    return;
  }
  if (hit >= 0) {
    selectedIndex = hit;
    dragging = true;
    brushRadius.value = obstacles[hit].r;
    updateBrushOutput();
  } else {
    const r = Number(brushRadius.value);
    obstacles.push({ x: clamp(point.x, r, WORLD - r), y: clamp(point.y, r, WORLD - r), r });
    selectedIndex = obstacles.length - 1;
    markDirty();
    draw();
  }
});

canvas.addEventListener("mousemove", (event) => {
  const point = canvasToWorld(event);
  pointerReadout.textContent = `x: ${point.x.toFixed(2)}, y: ${point.y.toFixed(2)}`;
  if (draggingPoint) {
    setSpecialPoint(draggingPoint, point);
    return;
  }
  if (!dragging || selectedIndex < 0) return;
  const obs = obstacles[selectedIndex];
  obs.x = clamp(point.x, obs.r, WORLD - obs.r);
  obs.y = clamp(point.y, obs.r, WORLD - obs.r);
  markDirty();
  draw();
});

window.addEventListener("mouseup", () => {
  dragging = false;
  draggingPoint = null;
});

canvas.addEventListener("contextmenu", (event) => event.preventDefault());
brushRadius.addEventListener("input", updateBrushOutput);

document.querySelectorAll(".mode").forEach((button) => {
  button.addEventListener("click", () => setMode(button.dataset.mode));
});

document.getElementById("deleteSelected").addEventListener("click", () => {
  if (selectedIndex >= 0) {
    obstacles.splice(selectedIndex, 1);
    selectedIndex = -1;
    markDirty();
    draw();
  }
});

document.getElementById("clearMap").addEventListener("click", () => {
  obstacles = [];
  selectedIndex = -1;
  markDirty("Harita temizlendi");
  draw();
});

document.getElementById("applyMapTemplate").addEventListener("click", () => {
  const kind = document.getElementById("mapTemplate").value;
  obstacles = templateObstacles(kind);
  startPoint = { ...DEFAULT_START };
  goalPoint = { ...DEFAULT_GOAL };
  selectedIndex = -1;
  if (kind !== "blank") {
    document.getElementById("mapName").value = `${kind}_custom`;
  }
  syncPointInputs();
  markDirty("Hazır harita yüklendi");
  draw();
});

document.getElementById("paramPreset").addEventListener("change", (event) => {
  setParams(presets[event.target.value]);
  markDirty("Setup uygulandı");
});

liquidCell.addEventListener("change", () => {
  markDirty("Liquid hücre tipi değişti");
});

document.getElementById("exportMap").addEventListener("click", () => {
  const data = currentMapData();
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${data.name || "custom_map"}.json`;
  link.click();
  URL.revokeObjectURL(url);
});

document.getElementById("importMap").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  const data = JSON.parse(await file.text());
  applyMapData(data);
  markDirty("JSON yüklendi");
});

function renderLinks(files) {
  const labels = {
    gif: "GIF",
    final_png: "Final PNG",
    map_png: "Harita PNG",
    config_json: "Config JSON",
  };
  outputLinks.innerHTML = "";
  Object.entries(files).forEach(([key, url]) => {
    if (!url) return;
    const link = document.createElement("a");
    link.href = `${url}?t=${Date.now()}`;
    link.target = "_blank";
    link.rel = "noreferrer";
    link.textContent = labels[key] || key;
    outputLinks.appendChild(link);
  });
}

function renderTraceTable(traces) {
  traceTable.innerHTML = "";
  traces.forEach((trace) => {
    const row = document.createElement("div");
    row.className = "trace-row";
    row.innerHTML = `
      <span><b>${trace.controller}</b><br>${trace.status}</span>
      <span>${trace.steps} adım<br>${trace.overrides} override</span>
      <span>${trace.dist_goal}<br>${trace.min_clearance}</span>
    `;
    traceTable.appendChild(row);
  });
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function setProgress(percent, message) {
  const value = Math.max(0, Math.min(100, Math.round(Number(percent) || 0)));
  progressWrap.hidden = false;
  progressBar.style.width = `${value}%`;
  progressText.textContent = `${value}%`;
  progressMessage.textContent = message || "Çalışıyor";
}

function hideProgressSoon() {
  window.setTimeout(() => {
    progressWrap.hidden = true;
  }, 1800);
}

function applySimulationResult(result) {
  runStatus.textContent = result.route_found
    ? `Kaydedildi: ${result.slug}`
    : `Kaydedildi: ${result.slug}. Planner rota bulamadı; yerel güvenlik kuralı kullanıldı.`;
  renderTraceTable(result.traces);
  renderLinks(result.files);
  const preview = result.files.gif || result.files.final_png || result.files.map_png;
  if (preview) {
    previewImage.src = `${preview}?t=${Date.now()}`;
    previewImage.style.display = "block";
  }
  saveState.textContent = "Kaydedildi";
}

async function refreshSavedMaps(selectSlug = "") {
  try {
    const response = await fetch("/api/maps");
    const result = await response.json();
    if (!result.ok) throw new Error(result.error || "Kayıtlı haritalar alınamadı.");
    savedMapSelect.innerHTML = "";
    if (!result.maps.length) {
      savedMapSelect.innerHTML = '<option value="">Kayıt yok</option>';
      mapLibraryStatus.textContent = "Henüz kayıtlı harita yok.";
      return;
    }
    result.maps.forEach((item) => {
      const option = document.createElement("option");
      option.value = item.slug;
      option.textContent = `${item.name} (${item.obstacles} engel)`;
      savedMapSelect.appendChild(option);
    });
    if (selectSlug) savedMapSelect.value = selectSlug;
    mapLibraryStatus.textContent = `${result.maps.length} kayıtlı harita`;
  } catch (error) {
    mapLibraryStatus.textContent = error.message;
  }
}

async function saveCurrentMap() {
  mapLibraryStatus.textContent = "Harita kaydediliyor.";
  try {
    const response = await fetch("/api/maps", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(currentMapData()),
    });
    const result = await response.json();
    if (!result.ok) throw new Error(result.error || "Harita kaydedilemedi.");
    await refreshSavedMaps(result.slug);
    saveState.textContent = "Harita kaydedildi";
    mapLibraryStatus.textContent = `Kaydedildi: ${result.name}`;
  } catch (error) {
    mapLibraryStatus.textContent = error.message;
  }
}

async function loadSelectedMap() {
  const slug = savedMapSelect.value;
  if (!slug) {
    mapLibraryStatus.textContent = "Yüklenecek harita seçilmedi.";
    return;
  }
  try {
    const response = await fetch(`/api/maps/${encodeURIComponent(slug)}`);
    const result = await response.json();
    if (!result.ok) throw new Error(result.error || "Harita yüklenemedi.");
    applyMapData(result.map);
    saveState.textContent = "Harita yüklendi";
    mapLibraryStatus.textContent = `Yüklendi: ${result.map.name}`;
  } catch (error) {
    mapLibraryStatus.textContent = error.message;
  }
}

async function deleteSelectedMap() {
  const slug = savedMapSelect.value;
  if (!slug) {
    mapLibraryStatus.textContent = "Silinecek harita seçilmedi.";
    return;
  }
  try {
    const response = await fetch(`/api/maps/${encodeURIComponent(slug)}`, { method: "DELETE" });
    const result = await response.json();
    if (!result.ok) throw new Error(result.error || "Harita silinemedi.");
    await refreshSavedMaps();
    saveState.textContent = "Harita silindi";
    mapLibraryStatus.textContent = `Silindi: ${slug}`;
  } catch (error) {
    mapLibraryStatus.textContent = error.message;
  }
}

async function runSimulation() {
  const payload = {
    ...currentMapData(),
  };
  runStatus.textContent = "Simülasyon başlatılıyor.";
  saveState.textContent = "Çalışıyor";
  setProgress(1, "İş başlatılıyor.");
  document.getElementById("runSimulation").disabled = true;
  document.getElementById("runTop").disabled = true;
  try {
    const response = await fetch("/api/simulate/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const started = await response.json();
    if (!started.ok) throw new Error(started.error || "Simülasyon başlatılamadı.");
    let job = null;
    while (true) {
      await sleep(650);
      const jobResponse = await fetch(`/api/jobs/${encodeURIComponent(started.job_id)}?t=${Date.now()}`);
      job = await jobResponse.json();
      if (!job.ok) throw new Error(job.error || "Simülasyon işi bulunamadı.");
      setProgress(job.progress || 0, job.message || "Çalışıyor");
      runStatus.textContent = job.message || "Simülasyon çalışıyor.";
      if (job.status === "done") {
        setProgress(100, "Tamamlandı.");
        applySimulationResult(job.result);
        hideProgressSoon();
        break;
      }
      if (job.status === "error") {
        throw new Error(job.error || job.message || "Simülasyon başarısız.");
      }
    }
  } catch (error) {
    runStatus.textContent = error.message;
    saveState.textContent = "Hata";
    setProgress(100, "Hata");
  } finally {
    document.getElementById("runSimulation").disabled = false;
    document.getElementById("runTop").disabled = false;
  }
}

document.getElementById("runSimulation").addEventListener("click", runSimulation);
document.getElementById("runTop").addEventListener("click", runSimulation);
document.getElementById("saveMap").addEventListener("click", saveCurrentMap);
document.getElementById("loadSavedMap").addEventListener("click", loadSelectedMap);
document.getElementById("deleteSavedMap").addEventListener("click", deleteSelectedMap);
["startX", "startY", "goalX", "goalY"].forEach((id) => {
  document.getElementById(id).addEventListener("change", readPointInputs);
});

buildControls();
setParams(presets.balanced);
syncPointInputs();
updateBrushOutput();
draw();
refreshSavedMaps();
