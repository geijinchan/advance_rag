/**
 * Cross-platform process manager for the Python FastAPI service.
 *
 * - Spawns uvicorn on port 3003
 * - Watches *.py under ./app and ./scripts, restarting the server on change
 * - Restarts on child crash with capped backoff
 * - Forwards stdout/stderr so logs are visible in the parent console
 *
 * Cross-platform: works on Windows, macOS, and Linux.
 */
const { spawn } = require("child_process");
const fs = require("fs");
const path = require("path");
const { execSync } = require("child_process");

const SERVICE_DIR = __dirname;
const PORT = 3003;
const WATCH_ROOTS = ["app", "scripts"];
const RESTART_DEBOUNCE_MS = 400;
const MAX_BACKOFF_MS = 8000;

/**
 * Detect the correct Python executable.
 * Priority: local .venv → system python3 → system python
 */
function findPython() {
  const isWindows = process.platform === "win32";
  const venvPy = isWindows
    ? path.join(SERVICE_DIR, ".venv", "Scripts", "python.exe")
    : path.join(SERVICE_DIR, ".venv", "bin", "python3");

  if (fs.existsSync(venvPy)) return venvPy;

  // Fallback: try python3 first (unix), then python (windows/anaconda)
  if (!isWindows) {
    try { execSync("python3 --version", { timeout: 3000 }); return "python3"; }
    catch (e) { /* not found */ }
  }
  return "python";
}

const PYTHON = findPython();

// --- hot-reload-safe singleton state ---------------------------------------
const G = globalThis;
if (!G.__ragAgentState) {
  G.__ragAgentState = {
    child: null,
    restartTimer: null,
    crashCount: 0,
    shuttingDown: false,
    starting: false,
    watchers: null,
    signalsBound: false,
  };
}
const state = G.__ragAgentState;

function log(tag, msg) {
  console.log(`[rag-agent:${tag}] ${msg}`);
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

/**
 * Cross-platform: find PIDs listening on PORT.
 * Uses `netstat` on Windows, `ss` or `lsof` on Unix.
 */
function foreignPortPids() {
  try {
    let out;
    const isWindows = process.platform === "win32";

    if (isWindows) {
      out = execSync(
        `netstat -ano | findstr "LISTENING" | findstr ":${PORT} "`,
        { timeout: 3000 }
      ).toString();
      const pids = [];
      for (const line of out.split("\n")) {
        const parts = line.trim().split(/\s+/);
        const pid = parseInt(parts[parts.length - 1], 10);
        if (!Number.isFinite(pid) || pid === 0) continue;
        if (state.child && pid === state.child.pid) continue;
        if (pid === process.pid) continue;
        pids.push(pid);
      }
      return [...new Set(pids)];
    } else {
      out = execSync(
        `ss -tlnp 2>/dev/null | grep 'LISTEN' | grep ':${PORT} ' || true`,
        { timeout: 3000 }
      ).toString();
      const pids = [];
      for (const m of out.matchAll(/pid=(\d+)/g)) {
        const pid = parseInt(m[1], 10);
        if (!Number.isFinite(pid)) continue;
        if (state.child && pid === state.child.pid) continue;
        if (pid === process.pid) continue;
        pids.push(pid);
      }
      return pids;
    }
  } catch (e) {
    return [];
  }
}

/** Kill foreign processes holding :PORT and wait for the port to free. */
async function freePort() {
  const isWindows = process.platform === "win32";
  for (let attempt = 0; attempt < 10; attempt++) {
    const pids = foreignPortPids();
    if (pids.length === 0) return true;
    for (const pid of pids) {
      try {
        if (isWindows) {
          execSync(`taskkill /F /PID ${pid}`, { timeout: 3000 });
        } else {
          process.kill(pid, "SIGTERM");
        }
        log("watcher", `port :${PORT} busy → killed orphan pid=${pid}`);
      } catch (e) { /* already dead */ }
    }
    await sleep(600);
  }
  return foreignPortPids().length === 0;
}

/** Stop the current child and wait until it is really gone. */
async function stopService(reason) {
  const c = state.child;
  if (!c || c.exitCode !== null) {
    state.child = null;
    return;
  }
  state.child = null;
  c.removeAllListeners("exit");
  const exited = new Promise((resolve) => c.once("exit", resolve));
  const isWindows = process.platform === "win32";

  if (isWindows) {
    try { execSync(`taskkill /F /PID ${c.pid} /T`, { timeout: 3000 }); } catch (e) { /* already dead */ }
  } else {
    try { c.kill("SIGTERM"); } catch (e) { /* already dead */ }
    const hardKill = setTimeout(() => {
      try { c.kill("SIGKILL"); } catch (e) { /* already dead */ }
    }, 1500);
    await Promise.race([exited, sleep(3000)]);
    clearTimeout(hardKill);
  }

  await Promise.race([exited, sleep(3000)]);
  log("watcher", `stopped previous uvicorn (pid=${c.pid}, ${reason})`);
}

async function startService() {
  if (state.starting) return;
  state.starting = true;
  try {
    await stopService("starting");
    await freePort();

    log("watcher", `using python: ${PYTHON}`);

    state.child = spawn(
      PYTHON,
      ["-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", String(PORT), "--log-level", "info"],
      {
        cwd: SERVICE_DIR,
        env: {
          ...process.env,
          PYTHONUNBUFFERED: "1",
        },
        // On Windows, use shell to handle .exe resolution from PATH
        shell: process.platform === "win32",
      }
    );

    state.child.stdout.on("data", (d) => process.stdout.write(d));
    state.child.stderr.on("data", (d) => process.stderr.write(d));

    state.child.on("exit", (code, signal) => {
      if (state.shuttingDown) return;
      state.crashCount += 1;
      const backoff = Math.min(500 * 2 ** (state.crashCount - 1), MAX_BACKOFF_MS);
      log("watcher", `uvicorn exited (code=${code} signal=${signal}); restarting in ${backoff}ms`);
      setTimeout(() => { startService().catch(() => {}); }, backoff);
    });

    log("watcher", `uvicorn started (pid=${state.child.pid}) on :${PORT}`);
  } catch (err) {
    log("watcher", `start failed: ${err && err.message}`);
  } finally {
    state.starting = false;
  }
}

function scheduleRestart(reason) {
  if (state.restartTimer) clearTimeout(state.restartTimer);
  state.restartTimer = setTimeout(() => {
    log("watcher", `restarting uvicorn (${reason})`);
    state.crashCount = 0;
    startService().catch(() => {});
  }, RESTART_DEBOUNCE_MS);
}

function watchPyFiles() {
  if (state.watchers) {
    log("watcher", "fs.watch already active — skipping re-registration");
    return;
  }
  const watchers = [];
  for (const root of WATCH_ROOTS) {
    const dir = path.join(SERVICE_DIR, root);
    if (!fs.existsSync(dir)) continue;
    try {
      const w = fs.watch(dir, { recursive: true }, (event, filename) => {
        if (filename && filename.toString().endsWith(".py")) {
          scheduleRestart(`file change: ${root}/${filename}`);
        }
      });
      watchers.push(w);
    } catch (err) {
      log("watcher", `fs.watch failed for ${dir}: ${err.message}`);
    }
  }
  state.watchers = watchers;
  log("watcher", `watching ${WATCH_ROOTS.join(", ")}/**/*.py for changes`);
}

function bindSignals() {
  if (state.signalsBound) return;
  state.signalsBound = true;
  const shutdown = (sig) => {
    state.shuttingDown = true;
    if (state.child) {
      if (process.platform === "win32") {
        try { execSync(`taskkill /F /PID ${state.child.pid} /T`, { timeout: 3000 }); } catch (e) {}
      } else {
        try { state.child.kill("SIGKILL"); } catch (e) {}
      }
    }
    process.exit(0);
  };
  process.on("SIGINT", () => shutdown("SIGINT"));
  process.on("SIGTERM", () => shutdown("SIGTERM"));
}

// Boot only if this is a genuine first start
if (!state.child && !state.starting) {
  startService().catch(() => {});
}
bindSignals();
watchPyFiles();
