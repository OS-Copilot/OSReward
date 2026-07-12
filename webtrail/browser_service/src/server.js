/**
 * webtrail browser service
 *
 * A small HTTP service that owns Playwright browser sessions on behalf of the
 * Python collector. One session == one isolated browsing state. The collector
 * talks to it with three verbs:
 *
 *   POST   /session                     create a session
 *   DELETE /session/:id                 destroy a session
 *   POST   /session/:id/goto            navigate
 *   POST   /session/:id/snapshot        screenshot + html + a11y tree + element map
 *   POST   /session/:id/act             execute one typed page action
 *   GET    /healthz                     liveness + session count
 *
 * Actions are typed and validated here (click / hover / scroll / type / press /
 * drag / select / check / back / forward / wait). The service never evaluates
 * caller-supplied code paths against the Playwright API.
 */

import express from "express";
import crypto from "crypto";
import { chromium } from "playwright-extra";
import StealthPlugin from "puppeteer-extra-plugin-stealth";

chromium.use(StealthPlugin());

// A single stray async rejection — most often a CDP or Playwright call landing
// after its page closed — must never take down the whole worker and every
// session running on it. Log and keep serving; per-request handlers already
// convert real failures into HTTP errors.
process.on("unhandledRejection", (reason) => {
  console.error("[unhandledRejection]", reason && reason.message || reason);
});
process.on("uncaughtException", (err) => {
  console.error("[uncaughtException]", err && err.message || err);
});

const PORT = parseInt(process.argv[2] || process.env.WEBTRAIL_PORT || "9300", 10);

const DEFAULTS = {
  width: 1920,
  height: 1080,
  navTimeoutMs: 45_000,
  actTimeoutMs: 15_000,
  settleMs: 800,
  netIdleMs: 2_500,
  htmlMaxBytes: 3_000_000,
  maxElements: 600,
  sessionTtlMs: 15 * 60 * 1000,
  reaperIntervalMs: 30 * 1000,
  typeDelayMs: 15,
  dragSteps: 16,
  maxWaitMs: 8_000,
};

// ---------------------------------------------------------------------------
// session registry
// ---------------------------------------------------------------------------

/** @type {Map<string, Session>} */
const REGISTRY = new Map();

let sharedBrowser = null; // lazily launched, used by "context" isolation mode
let sharedBrowserContexts = 0;
const SHARED_BROWSER_RECYCLE_AFTER = 48; // contexts served before relaunch

class Session {
  constructor({ id, browser, context, page, ownsBrowser, viewport }) {
    this.id = id;
    this.browser = browser;
    this.context = context;
    this.page = page;
    this.ownsBrowser = ownsBrowser;
    this.viewport = viewport;
    this.createdAt = Date.now();
    this.lastUsed = Date.now();
    this.closing = false;
    this.cdp = null; // lazily attached, used for the accessibility tree
  }

  touch() {
    this.lastUsed = Date.now();
  }

  async dispose() {
    if (this.closing) return;
    this.closing = true;
    try { await this.context.close(); } catch {}
    if (this.ownsBrowser) {
      try { await this.browser.close(); } catch {}
    }
  }
}

function launchArgs() {
  const args = [
    "--disable-blink-features=AutomationControlled",
    "--disable-dev-shm-usage",     // small /dev/shm in containers crashes tabs
    "--no-first-run",
    "--mute-audio",
  ];
  // The Chromium sandbox cannot start inside most Linux containers (and never
  // as root), where it aborts the whole launch. Disable it on Linux by default;
  // set WEBTRAIL_SANDBOX=1 to force it back on where the host supports it.
  if (process.platform === "linux" && process.env.WEBTRAIL_SANDBOX !== "1") {
    args.push("--no-sandbox", "--disable-setuid-sandbox");
  }
  // headless GPU stack is frequently absent on servers; software rendering is
  // more reliable for screenshots
  if (process.platform === "linux") {
    args.push("--disable-gpu", "--disable-software-rasterizer");
  }
  if (process.env.WEBTRAIL_CHROMIUM_ARGS) {
    args.push(...process.env.WEBTRAIL_CHROMIUM_ARGS.split(/\s+/).filter(Boolean));
  }
  return args;
}

async function getSharedBrowser() {
  const alive = sharedBrowser && sharedBrowser.isConnected();
  if (!alive || sharedBrowserContexts >= SHARED_BROWSER_RECYCLE_AFTER) {
    if (alive) {
      const old = sharedBrowser;
      // let existing contexts finish; close once orphaned
      setTimeout(() => { old.close().catch(() => {}); }, DEFAULTS.sessionTtlMs);
    }
    sharedBrowser = await chromium.launch({ headless: true, args: launchArgs() });
    sharedBrowserContexts = 0;
  }
  sharedBrowserContexts += 1;
  return sharedBrowser;
}

/**
 * Keep navigation inside a single tab: force target=_blank links and
 * window.open() calls to reuse the current tab so the trajectory stays linear.
 */
const SINGLE_TAB_INIT = `
  (() => {
    const open = window.open;
    window.open = function (url) {
      if (url) { window.location.href = url; return window; }
      return open.apply(window, arguments);
    };
    addEventListener("click", (ev) => {
      const link = ev.target && ev.target.closest ? ev.target.closest("a[target]") : null;
      if (link && link.target && link.target !== "_self") link.target = "_self";
    }, true);
  })();
`;

async function createSession(opts) {
  const width = opts.width || DEFAULTS.width;
  const height = opts.height || DEFAULTS.height;
  const isolation = opts.isolation === "context" ? "context" : "browser";

  const contextOptions = {
    viewport: { width, height },
    deviceScaleFactor: 1,
    locale: opts.locale || "en-US",
    timezoneId: opts.timezone || undefined,
    userAgent: opts.userAgent || undefined,
    extraHTTPHeaders: opts.extraHeaders || undefined,
    proxy: opts.proxy || undefined,
    ignoreHTTPSErrors: true,
  };

  let browser;
  let ownsBrowser;
  if (isolation === "browser") {
    browser = await chromium.launch({
      headless: true,
      args: launchArgs(),
      proxy: opts.proxy || undefined,
    });
    ownsBrowser = true;
  } else {
    browser = await getSharedBrowser();
    ownsBrowser = false;
  }

  let context;
  try {
    context = await browser.newContext(contextOptions);
  } catch (err) {
    if (ownsBrowser) await browser.close().catch(() => {});
    throw err;
  }

  await context.addInitScript(SINGLE_TAB_INIT);
  context.setDefaultNavigationTimeout(opts.navTimeoutMs || DEFAULTS.navTimeoutMs);
  context.setDefaultTimeout(opts.actTimeoutMs || DEFAULTS.actTimeoutMs);

  const page = await context.newPage();
  // headless pages start without frame focus, which splits keyboard handling
  // (chars reach the focused element but navigation keys scroll the page)
  await page.bringToFront().catch(() => {});

  // safety net: if a popup slips through, adopt nothing and close it
  context.on("page", (extra) => {
    if (extra !== page) extra.close().catch(() => {});
  });

  const session = new Session({
    id: crypto.randomUUID(),
    browser,
    context,
    page,
    ownsBrowser,
    viewport: { width, height },
  });
  REGISTRY.set(session.id, session);
  return session;
}

// ---------------------------------------------------------------------------
// page snapshot
// ---------------------------------------------------------------------------

/**
 * Runs inside the page. Collects document HTML (with same-origin iframe bodies
 * inlined as comments), a compact map of visible interactive elements, and
 * scroll geometry.
 */
function pageCollector({ htmlMaxBytes, maxElements }) {
  const interactiveSelector = [
    "a[href]", "button", "input", "select", "textarea", "summary",
    "[role]", "[onclick]", "[contenteditable='true']", "[tabindex]",
  ].join(",");

  const accessibleName = (el) => {
    const aria = el.getAttribute("aria-label");
    if (aria) return aria;
    const labelledBy = el.getAttribute("aria-labelledby");
    if (labelledBy) {
      const parts = labelledBy.split(/\s+/)
        .map((id) => document.getElementById(id))
        .filter(Boolean)
        .map((n) => n.textContent.trim());
      if (parts.length) return parts.join(" ");
    }
    if (el.labels && el.labels.length) return el.labels[0].textContent.trim();
    const placeholder = el.getAttribute("placeholder");
    if (placeholder) return placeholder;
    const title = el.getAttribute("title");
    if (title) return title;
    const alt = el.getAttribute("alt");
    if (alt) return alt;
    return (el.innerText || el.value || "").trim();
  };

  const impliedRole = (el) => {
    const explicit = el.getAttribute("role");
    if (explicit) return explicit;
    const tag = el.tagName.toLowerCase();
    if (tag === "a") return "link";
    if (tag === "button" || tag === "summary") return "button";
    if (tag === "select") return "combobox";
    if (tag === "textarea") return "textbox";
    if (tag === "input") {
      const type = (el.getAttribute("type") || "text").toLowerCase();
      if (type === "checkbox" || type === "radio" || type === "button" ||
          type === "submit" || type === "range" || type === "search") return type;
      return "textbox";
    }
    return tag;
  };

  const elements = [];
  const seen = new Set();
  for (const el of document.querySelectorAll(interactiveSelector)) {
    if (elements.length >= maxElements) break;
    if (seen.has(el)) continue;
    seen.add(el);
    const rect = el.getBoundingClientRect();
    if (rect.width < 1 || rect.height < 1) continue;
    let visible = true;
    try { visible = el.checkVisibility({ opacityProperty: true, visibilityProperty: true }); } catch {}
    if (!visible) continue;

    const record = {
      role: impliedRole(el),
      name: accessibleName(el).slice(0, 120),
      bbox: [
        Math.round(rect.x), Math.round(rect.y),
        Math.round(rect.width), Math.round(rect.height),
      ],
      in_viewport: rect.bottom > 0 && rect.right > 0 &&
        rect.top < window.innerHeight && rect.left < window.innerWidth,
    };
    if ((el.tagName === "INPUT" && el.type !== "checkbox" && el.type !== "radio") ||
        el.tagName === "TEXTAREA") {
      record.value = String(el.value).slice(0, 200);
    }
    if (el.tagName === "SELECT") {
      record.value = el.selectedIndex >= 0 && el.options[el.selectedIndex]
        ? el.options[el.selectedIndex].text.slice(0, 120) : null;
    }
    if (el.type === "checkbox" || el.type === "radio") record.checked = el.checked;
    if (el.disabled) record.disabled = true;
    if (document.activeElement === el) record.focused = true;
    if (el.tagName === "A") {
      const href = el.getAttribute("href");
      if (href && !href.startsWith("javascript:")) record.href = href.slice(0, 300);
    }
    elements.push(record);
  }

  let html = "<!DOCTYPE html>\n" + document.documentElement.outerHTML;
  const frameParts = [];
  for (const frame of document.querySelectorAll("iframe")) {
    const src = frame.getAttribute("src") || "";
    try {
      const doc = frame.contentDocument;
      if (doc && doc.documentElement) {
        frameParts.push(
          `\n<!-- embedded frame ${src} -->\n` + doc.documentElement.outerHTML
        );
      }
    } catch {
      /* cross-origin frame, skip */
    }
  }
  if (frameParts.length) html += frameParts.join("");
  if (html.length > htmlMaxBytes) html = html.slice(0, htmlMaxBytes);

  return {
    html,
    elements,
    title: document.title,
    scroll: {
      x: Math.round(window.scrollX),
      y: Math.round(window.scrollY),
      page_height: Math.round(
        Math.max(document.body ? document.body.scrollHeight : 0,
                 document.documentElement.scrollHeight)
      ),
      viewport_height: window.innerHeight,
    },
  };
}

async function takeSnapshot(session, opts) {
  const page = session.page;
  const want = {
    screenshot: opts.screenshot !== false,
    html: opts.html !== false,
    axtree: opts.axtree !== false,
    elements: opts.elements !== false,
  };

  // let the page settle before observing: DOM ready, then a fixed pause,
  // then best-effort network idle so slow XHR-rendered pages have a chance
  try { await page.waitForLoadState("domcontentloaded", { timeout: 10_000 }); } catch {}
  const settleMs = opts.settleMs ?? DEFAULTS.settleMs;
  if (settleMs > 0) await page.waitForTimeout(settleMs);
  try {
    await page.waitForLoadState("networkidle", { timeout: opts.netIdleMs ?? DEFAULTS.netIdleMs });
  } catch {}

  const result = {
    url: page.url(),
    viewport: session.viewport,
    taken_at: Date.now(),
  };

  if (want.html || want.elements) {
    try {
      const collected = await page.evaluate(pageCollector, {
        htmlMaxBytes: opts.htmlMaxBytes ?? DEFAULTS.htmlMaxBytes,
        maxElements: opts.maxElements ?? DEFAULTS.maxElements,
      });
      result.title = collected.title;
      result.scroll = collected.scroll;
      if (want.html) result.html = collected.html;
      if (want.elements) result.elements = collected.elements;
    } catch (err) {
      result.collect_error = String(err && err.message || err);
    }
  }

  if (result.title === undefined) {
    try { result.title = await page.title(); } catch { result.title = null; }
  }

  if (want.axtree) {
    try {
      result.axtree = await accessibilityTree(session);
    } catch (err) {
      result.axtree = null;
      result.axtree_error = String(err && err.message || err);
      session.cdp = null; // stale after navigation-induced detach; re-attach next time
    }
  }

  if (want.screenshot) {
    try {
      const buffer = await page.screenshot({
        type: "png",
        timeout: opts.screenshotTimeoutMs ?? DEFAULTS.actTimeoutMs,
        animations: "disabled",
      });
      result.screenshot = buffer.toString("base64");
    } catch (err) {
      // pages with perpetual animations can stall page.screenshot; raw CDP
      // capture does not wait for compositor stability and usually succeeds.
      // A wedged renderer can hang CDP too, so the fallback is race-bounded.
      try {
        if (!session.cdp) {
          session.cdp = await session.context.newCDPSession(page);
        }
        // the losing side of the race must keep a catch attached, or a late
        // rejection (page closed after timeout) escapes as unhandledRejection
        const cdpShot = session.cdp.send("Page.captureScreenshot", { format: "png" });
        cdpShot.catch(() => {});
        const shot = await Promise.race([
          cdpShot,
          new Promise((_, reject) => setTimeout(
            () => reject(new Error("cdp screenshot timeout")), 8_000)),
        ]);
        result.screenshot = shot.data;
        result.screenshot_note = "cdp_fallback";
      } catch (err2) {
        result.screenshot = null;
        result.screenshot_error = String(err && err.message || err);
        session.cdp = null;
      }
    }
  }

  return result;
}

/**
 * Accessibility tree via CDP, reduced to the nodes that matter to an agent:
 * named nodes and interactive/structural roles. Uninteresting wrappers are
 * collapsed so their children float up, mirroring the classic
 * "interesting-only" snapshot.
 */
const AX_KEEP_ROLES = new Set([
  "button", "link", "textbox", "searchbox", "combobox", "listbox", "checkbox",
  "radio", "switch", "slider", "spinbutton", "menuitem", "menuitemcheckbox",
  "menuitemradio", "tab", "option", "heading", "img", "image", "dialog",
  "alert", "alertdialog", "form", "table", "grid", "navigation", "main",
  "banner", "search", "contentinfo", "article", "region", "list", "listitem",
]);
const AX_MAX_NODES = 2500;

async function accessibilityTree(session) {
  if (!session.cdp) {
    session.cdp = await session.context.newCDPSession(session.page);
  }
  const { nodes } = await session.cdp.send("Accessibility.getFullAXTree");
  const byId = new Map(nodes.map((n) => [n.nodeId, n]));
  let kept = 0;

  const build = (nodeId) => {
    const node = byId.get(nodeId);
    if (!node || kept >= AX_MAX_NODES) return [];
    const role = node.role ? String(node.role.value) : "";
    if (role === "InlineTextBox") return []; // layout fragments, pure noise
    let children = (node.childIds || []).flatMap(build);
    if (node.ignored) return children;

    const name = node.name ? String(node.name.value).slice(0, 160) : "";
    const interesting = (name && role !== "generic" && role !== "none") ||
      AX_KEEP_ROLES.has(role);
    if (!interesting) return children;

    // a leaf StaticText that merely repeats its parent's name adds nothing
    children = children.filter((c) =>
      !(c.role === "StaticText" && c.name === name && !c.children));

    kept += 1;
    const entry = { role, name };
    if (node.value && node.value.value !== undefined && node.value.value !== "") {
      entry.value = String(node.value.value).slice(0, 200);
    }
    for (const prop of node.properties || []) {
      if (["checked", "expanded", "selected", "disabled", "focused", "pressed",
           "level"].includes(prop.name) && prop.value) {
        entry[prop.name] = prop.value.value;
      }
    }
    if (children.length) entry.children = children;
    return [entry];
  };

  const rootId = nodes.length ? nodes[0].nodeId : null;
  const tree = rootId === null ? [] : build(rootId);
  return tree.length === 1 ? tree[0] : { role: "document", children: tree };
}

// ---------------------------------------------------------------------------
// typed actions
// ---------------------------------------------------------------------------

const SELECT_ALL = process.platform === "darwin" ? "Meta+A" : "Control+A";

function numberPair(body, xKey = "x", yKey = "y") {
  const x = Number(body[xKey]);
  const y = Number(body[yKey]);
  if (!Number.isFinite(x) || !Number.isFinite(y)) {
    throw new Error(`action requires numeric ${xKey}/${yKey}`);
  }
  return [x, y];
}

/** Resolve the element under a point, walking into open shadow roots. */
function deepElementScript([x, y]) {
  let node = document.elementFromPoint(x, y);
  let inner = node && node.shadowRoot ? node.shadowRoot.elementFromPoint(x, y) : null;
  while (inner && inner !== node) {
    node = inner;
    inner = node.shadowRoot ? node.shadowRoot.elementFromPoint(x, y) : null;
  }
  return node;
}

const ACTION_HANDLERS = {
  async click(page, body) {
    const [x, y] = numberPair(body);
    await page.mouse.click(x, y, {
      button: body.button === "right" ? "right" : "left",
      clickCount: body.count === 2 ? 2 : 1,
    });
  },

  async hover(page, body) {
    const [x, y] = numberPair(body);
    await page.mouse.move(x, y);
  },

  async scroll(page, body) {
    const dx = Number(body.dx) || 0;
    const dy = Number(body.dy) || 0;
    if (dx === 0 && dy === 0) throw new Error("scroll requires non-zero dx or dy");
    if (Number.isFinite(Number(body.x)) && Number.isFinite(Number(body.y))) {
      await page.mouse.move(Number(body.x), Number(body.y));
    }
    await page.mouse.wheel(dx, dy);
  },

  async type(page, body) {
    if (typeof body.text !== "string") throw new Error("type requires text");
    if (Number.isFinite(Number(body.x)) && Number.isFinite(Number(body.y))) {
      await page.mouse.click(Number(body.x), Number(body.y));
      await page.waitForTimeout(120);
    }
    if (body.clear) {
      await page.keyboard.press(SELECT_ALL);
      await page.keyboard.press("Backspace");
    }
    await page.keyboard.type(body.text, { delay: body.delayMs ?? DEFAULTS.typeDelayMs });
    if (body.enter) await page.keyboard.press("Enter");
  },

  async press(page, body) {
    let combos = body.keys;
    if (typeof combos === "string") combos = [combos];
    if (!Array.isArray(combos) || combos.length === 0 || combos.length > 24) {
      throw new Error("press requires keys: string or array of up to 24 combos");
    }
    const repeat = Math.min(Math.max(Number(body.repeat) || 1, 1), 60);
    for (let round = 0; round < repeat; round++) {
      for (const combo of combos) {
        if (typeof combo !== "string" || !combo.length) throw new Error("invalid key combo");
        await page.keyboard.press(combo);
      }
    }
  },

  async drag(page, body) {
    const [x1, y1] = numberPair(body, "x1", "y1");
    const [x2, y2] = numberPair(body, "x2", "y2");
    const steps = Math.min(Math.max(Number(body.steps) || DEFAULTS.dragSteps, 2), 60);
    await page.mouse.move(x1, y1);
    await page.mouse.down();
    await page.mouse.move(x2, y2, { steps });
    await page.mouse.up();
  },

  async select(page, body) {
    const [x, y] = numberPair(body);
    const handle = await page.evaluateHandle(deepElementScript, [x, y]);
    const element = handle.asElement();
    let done = false;
    if (element) {
      const target = await element.evaluateHandle((node) => node.closest("select"));
      const selectEl = target.asElement();
      if (selectEl) {
        if (body.label !== undefined) {
          await selectEl.selectOption({ label: String(body.label) });
        } else if (body.value !== undefined) {
          await selectEl.selectOption(String(body.value));
        } else if (body.index !== undefined) {
          await selectEl.selectOption({ index: Number(body.index) });
        } else {
          throw new Error("select requires label, value, or index");
        }
        done = true;
      }
      await target.dispose().catch(() => {});
    }
    await handle.dispose().catch(() => {});
    // no native <select> under the point: it is a custom widget, open it
    if (!done) await page.mouse.click(x, y);
  },

  async check(page, body) {
    const [x, y] = numberPair(body);
    const desired = body.checked !== false;
    const handle = await page.evaluateHandle(deepElementScript, [x, y]);
    const element = handle.asElement();
    let done = false;
    if (element) {
      try {
        const kind = await element.evaluate((node) =>
          node.tagName === "INPUT" ? node.type : null);
        if (kind === "checkbox" || kind === "radio") {
          await element.setChecked(desired, { force: true });
          done = true;
        }
      } catch {}
    }
    await handle.dispose().catch(() => {});
    if (!done) await page.mouse.click(x, y);
  },

  async back(page) {
    await page.goBack({ waitUntil: "domcontentloaded" }).catch(() => {});
  },

  async forward(page) {
    await page.goForward({ waitUntil: "domcontentloaded" }).catch(() => {});
  },

  async wait(page, body) {
    const ms = Math.min(Math.max(Number(body.ms) || 1000, 50), DEFAULTS.maxWaitMs);
    await page.waitForTimeout(ms);
  },
};

// ---------------------------------------------------------------------------
// HTTP layer
// ---------------------------------------------------------------------------

const app = express();
app.use(express.json({ limit: "2mb" }));

function getSession(req, res) {
  const session = REGISTRY.get(req.params.id);
  if (!session || session.closing) {
    res.status(410).json({ ok: false, error: "unknown_session" });
    return null;
  }
  session.touch();
  return session;
}

app.post("/session", async (req, res) => {
  try {
    const session = await createSession(req.body || {});
    res.json({
      ok: true,
      session_id: session.id,
      viewport: session.viewport,
    });
  } catch (err) {
    res.status(500).json({ ok: false, error: `session_start_failed: ${err.message || err}` });
  }
});

app.delete("/session/:id", async (req, res) => {
  const session = REGISTRY.get(req.params.id);
  if (session) {
    REGISTRY.delete(session.id);
    await session.dispose();
  }
  res.json({ ok: true });
});

app.post("/session/:id/goto", async (req, res) => {
  const session = getSession(req, res);
  if (!session) return;
  let { url, timeoutMs, waitUntil } = req.body || {};
  if (typeof url !== "string" || !url) {
    return res.status(400).json({ ok: false, error: "url required" });
  }
  try {
    // agents frequently emit relative links ("/search?q=x"); resolve them
    // against the page they are looking at
    url = new URL(url, session.page.url() || undefined).toString();
  } catch {
    return res.json({ ok: false, error: `goto_failed: unresolvable url ${url}` });
  }
  try {
    const response = await session.page.goto(url, {
      timeout: timeoutMs || DEFAULTS.navTimeoutMs,
      waitUntil: waitUntil || "domcontentloaded",
    });
    res.json({
      ok: true,
      status: response ? response.status() : null,
      final_url: session.page.url(),
    });
  } catch (err) {
    res.json({ ok: false, error: `goto_failed: ${err.message || err}`, final_url: safeUrl(session) });
  }
});

app.post("/session/:id/snapshot", async (req, res) => {
  const session = getSession(req, res);
  if (!session) return;
  try {
    const snapshot = await takeSnapshot(session, req.body || {});
    res.json({ ok: true, ...snapshot });
  } catch (err) {
    res.status(500).json({ ok: false, error: `snapshot_failed: ${err.message || err}` });
  }
});

app.post("/session/:id/act", async (req, res) => {
  const session = getSession(req, res);
  if (!session) return;
  const body = req.body || {};
  const handler = ACTION_HANDLERS[body.kind];
  if (!handler) {
    return res.status(400).json({ ok: false, error: `unknown action kind: ${body.kind}` });
  }
  const startedAt = Date.now();
  try {
    await handler(session.page, body);
    res.json({ ok: true, elapsed_ms: Date.now() - startedAt, final_url: safeUrl(session) });
  } catch (err) {
    res.json({
      ok: false,
      error: `${body.kind}_failed: ${err.message || err}`,
      elapsed_ms: Date.now() - startedAt,
      final_url: safeUrl(session),
    });
  }
});

app.get("/healthz", (_req, res) => {
  res.json({
    ok: true,
    port: PORT,
    sessions: REGISTRY.size,
    rss_mb: Math.round(process.memoryUsage().rss / 1024 / 1024),
    uptime_s: Math.round(process.uptime()),
  });
});

function safeUrl(session) {
  try { return session.page.url(); } catch { return null; }
}

// ---------------------------------------------------------------------------
// lifecycle
// ---------------------------------------------------------------------------

setInterval(async () => {
  const now = Date.now();
  for (const [id, session] of REGISTRY) {
    if (now - session.lastUsed > DEFAULTS.sessionTtlMs) {
      REGISTRY.delete(id);
      console.log(`[reaper] closing idle session ${id.slice(0, 8)}`);
      await session.dispose();
    }
  }
}, DEFAULTS.reaperIntervalMs);

async function shutdown() {
  console.log("shutting down, closing sessions...");
  const pending = [...REGISTRY.values()].map((s) => s.dispose());
  REGISTRY.clear();
  await Promise.allSettled(pending);
  if (sharedBrowser) await sharedBrowser.close().catch(() => {});
  process.exit(0);
}

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);

app.listen(PORT, () => {
  console.log(`webtrail browser service listening on :${PORT}`);
});
