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

const stealth = StealthPlugin();
// This evasion proxies iframe.contentWindow by monkey-patching
// document.createElement.  Some real sites (notably Lawyerist) use iframe
// measurements during layout; the proxy makes their code set body width to
// 4000px inside a 1920px viewport, shifting most visible content off-screen.
// Keep every other stealth evasion while preserving native iframe geometry.
stealth.enabledEvasions.delete("iframe.contentWindow");
chromium.use(stealth);

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
  snapshotTimeoutMs: 45_000,
  domCollectTimeoutMs: 8_000,
  axtreeTimeoutMs: 8_000,
  cdpScreenshotTimeoutMs: 8_000,
  playwrightScreenshotTimeoutMs: 12_000,
  titleTimeoutMs: 2_000,
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

function withDeadline(promise, timeoutMs, label) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(
      () => reject(new Error(`${label} timed out after ${timeoutMs}ms`)),
      timeoutMs,
    );
  });
  // Promise.race attaches rejection handlers to every input, so a Playwright
  // operation rejecting after the deadline cannot become an unhandled
  // rejection. Playwright does not expose cancellation for these calls.
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

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
    this.mainDocumentStatus = null;
    this.mainDocumentUrl = null;
  }

  touch() {
    this.lastUsed = Date.now();
  }

  recordMainDocumentResponse(url, status) {
    this.mainDocumentUrl = url || null;
    this.mainDocumentStatus = Number.isInteger(status) ? status : null;
  }

  clearMainDocumentResponse() {
    this.mainDocumentUrl = null;
    this.mainDocumentStatus = null;
  }

  currentHttpStatus() {
    if (!this.mainDocumentUrl || this.mainDocumentStatus === null) return null;
    try {
      const current = new URL(this.page.url());
      const observed = new URL(this.mainDocumentUrl);
      current.hash = "";
      observed.hash = "";
      return current.href === observed.href ? this.mainDocumentStatus : null;
    } catch {
      return null;
    }
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
  // Do not disable both GPU compositing and Chromium's software rasterizer.
  // Headless Chromium automatically chooses an available GPU or its software
  // fallback; disabling both made renderer/compositor screenshot stalls much
  // more likely on GPU-less collection hosts. Operators can still supply a
  // host-specific GL backend through WEBTRAIL_CHROMIUM_ARGS.
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
  page.on("response", (response) => {
    try {
      if (response.frame() === page.mainFrame() &&
          response.request().isNavigationRequest()) {
        session.recordMainDocumentResponse(response.url(), response.status());
      }
    } catch {
      // The page may close while an event is being delivered. The next
      // snapshot will simply carry a null status instead of stale metadata.
    }
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
    http_status: session.currentHttpStatus(),
    viewport: session.viewport,
    taken_at: Date.now(),
    snapshot_stage_ms: {},
  };

  const stage = async (name, promise, timeoutMs) => {
    const startedAt = Date.now();
    try {
      return await withDeadline(promise, timeoutMs, name);
    } finally {
      result.snapshot_stage_ms[name] = Date.now() - startedAt;
    }
  };

  // Capture the viewport before DOM/AX extraction. CDP does not wait for web
  // fonts or animation stability, avoiding Playwright's most common screenshot
  // stall. A dedicated CDP session isolates capture from a stuck AX request.
  if (want.screenshot) {
    let cdp;
    let cdpError;
    try {
      cdp = await stage(
        "cdp_attach",
        session.context.newCDPSession(page),
        2_000,
      );
      const shot = await stage(
        "cdp_screenshot",
        cdp.send("Page.captureScreenshot", {
          format: "png",
          fromSurface: true,
          captureBeyondViewport: false,
        }),
        opts.cdpScreenshotTimeoutMs ?? DEFAULTS.cdpScreenshotTimeoutMs,
      );
      result.screenshot = shot.data;
      result.screenshot_note = "cdp_viewport";
    } catch (err) {
      cdpError = err;
    } finally {
      if (cdp) cdp.detach().catch(() => {});
    }

    if (!result.screenshot) {
      try {
        const buffer = await stage(
          "playwright_screenshot",
          page.screenshot({
            type: "png",
            timeout: opts.screenshotTimeoutMs
              ?? DEFAULTS.playwrightScreenshotTimeoutMs,
            animations: "disabled",
          }),
          (opts.screenshotTimeoutMs ?? DEFAULTS.playwrightScreenshotTimeoutMs) + 1_000,
        );
        result.screenshot = buffer.toString("base64");
        result.screenshot_note = "playwright_fallback";
        // The primary compositor path was unhealthy. Return the useful image
        // without risking another long DOM/AX call on the same renderer.
        result.snapshot_degraded = true;
      } catch (err) {
        result.screenshot = null;
        result.screenshot_error = [
          `cdp: ${String(cdpError && cdpError.message || cdpError)}`,
          `playwright: ${String(err && err.message || err)}`,
        ].join("; ");
        // The collector cannot use an observation without an image and will
        // retry it. Avoid spending more time collecting DOM/AX for this attempt.
        return result;
      }
    }
  }

  if (result.snapshot_degraded) {
    try {
      result.title = await stage(
        "title",
        page.title(),
        opts.titleTimeoutMs ?? DEFAULTS.titleTimeoutMs,
      );
    } catch {
      result.title = null;
    }
    return result;
  }

  if (want.html || want.elements) {
    try {
      const collected = await stage(
        "dom_collect",
        page.evaluate(pageCollector, {
          htmlMaxBytes: opts.htmlMaxBytes ?? DEFAULTS.htmlMaxBytes,
          maxElements: opts.maxElements ?? DEFAULTS.maxElements,
        }),
        opts.domCollectTimeoutMs ?? DEFAULTS.domCollectTimeoutMs,
      );
      result.title = collected.title;
      result.scroll = collected.scroll;
      if (want.html) result.html = collected.html;
      if (want.elements) result.elements = collected.elements;
    } catch (err) {
      result.collect_error = String(err && err.message || err);
    }
  }

  if (result.title === undefined) {
    try {
      result.title = await stage(
        "title",
        page.title(),
        opts.titleTimeoutMs ?? DEFAULTS.titleTimeoutMs,
      );
    } catch {
      result.title = null;
    }
  }

  if (want.axtree) {
    try {
      result.axtree = await stage(
        "axtree",
        accessibilityTree(session),
        opts.axtreeTimeoutMs ?? DEFAULTS.axtreeTimeoutMs,
      );
    } catch (err) {
      result.axtree = null;
      result.axtree_error = String(err && err.message || err);
      const staleCdp = session.cdp;
      session.cdp = null; // stale or blocked; re-attach next time
      if (staleCdp) staleCdp.detach().catch(() => {});
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

/**
 * Resolve a top-level screenshot coordinate to the deepest element under it.
 * Playwright's element bounding boxes use main-frame viewport coordinates, so
 * the same global point can be translated into each child frame without DOM
 * selectors or model-visible element IDs.  This also works for cross-origin
 * frames because each evaluation runs in that frame's own execution context.
 */
async function elementAtPagePoint(page, x, y) {
  let frame = page.mainFrame();
  let localX = x;
  let localY = y;
  let offsetX = 0;
  let offsetY = 0;

  for (let depth = 0; depth < 8; depth += 1) {
    const handle = await frame.evaluateHandle(deepElementScript, [localX, localY]);
    const element = handle.asElement();
    if (!element) {
      await handle.dispose().catch(() => {});
      return { frame, element: null, x: localX, y: localY, offsetX, offsetY };
    }

    const tagName = await element.evaluate((node) => node.tagName).catch(() => "");
    if (tagName !== "IFRAME" && tagName !== "FRAME") {
      return { frame, element, x: localX, y: localY, offsetX, offsetY };
    }

    const child = await element.contentFrame().catch(() => null);
    const box = await element.boundingBox().catch(() => null);
    const border = await element.evaluate((node) => ({
      left: Number(node.clientLeft) || 0,
      top: Number(node.clientTop) || 0,
    })).catch(() => ({ left: 0, top: 0 }));
    if (!child || !box) {
      return { frame, element, x: localX, y: localY, offsetX, offsetY };
    }

    await handle.dispose().catch(() => {});
    offsetX = box.x + border.left;
    offsetY = box.y + border.top;
    localX = x - offsetX;
    localY = y - offsetY;
    frame = child;
  }

  return { frame, element: null, x: localX, y: localY, offsetX, offsetY };
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
    const hasPoint = Number.isFinite(Number(body.x)) && Number.isFinite(Number(body.y));
    const x = hasPoint ? Number(body.x) : null;
    const y = hasPoint ? Number(body.y) : null;
    const readEditable = async () => page.evaluate(([px, py]) => {
      let node = Number.isFinite(px) && Number.isFinite(py)
        ? document.elementFromPoint(px, py)
        : document.activeElement;
      if (node && node.shadowRoot && Number.isFinite(px) && Number.isFinite(py)) {
        let inner = node.shadowRoot.elementFromPoint(px, py);
        while (inner && inner !== node) {
          node = inner;
          inner = node.shadowRoot ? node.shadowRoot.elementFromPoint(px, py) : null;
        }
      }
      const editable = node && node.closest
        ? node.closest('input, textarea, [contenteditable="true"]')
        : null;
      if (!editable) return null;
      const isPassword = editable.tagName === "INPUT" && editable.type === "password";
      const value = editable.isContentEditable ? editable.textContent : editable.value;
      return { value: String(value ?? ""), is_password: isPassword };
    }, [x, y]).catch(() => null);

    const before = await readEditable();
    if (hasPoint) {
      await page.mouse.click(x, y);
      await page.waitForTimeout(120);
    }
    if (body.clear) {
      await page.keyboard.press(SELECT_ALL);
      await page.keyboard.press("Backspace");
    }
    await page.keyboard.type(body.text, { delay: body.delayMs ?? DEFAULTS.typeDelayMs });
    if (body.enter) await page.keyboard.press("Enter");
    await page.waitForTimeout(50).catch(() => {});

    const after = await readEditable();
    if (!after || after.is_password) return {};
    const expected = body.clear ? body.text : `${before && !before.is_password ? before.value : ""}${body.text}`;
    return {
      actual_value: after.value,
      value_matches: after.value === expected,
    };
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
    const hit = await elementAtPagePoint(page, x, y);
    const element = hit.element;
    if (body.label === undefined && body.value === undefined) {
      if (element) await element.dispose().catch(() => {});
      throw new Error("select requires the visible option label or value");
    }

    // Browser-use first handles a native <select>, matching text/value without
    // relying on the model to know an element index.
    if (element) {
      const target = await element.evaluateHandle((node) => node.closest("select"));
      const selectEl = target.asElement();
      if (selectEl) {
        try {
          const options = await selectEl.evaluate((select) => Array.from(select.options).map((option, index) => ({
            index,
            label: option.text.trim(),
            value: option.value,
          })));
          const needle = String(body.label !== undefined ? body.label : body.value).trim().toLowerCase();
          const wanted = options.find((option) =>
            option.label.toLowerCase() === needle || option.value.toLowerCase() === needle
          ) || null;
          if (!wanted) {
            return {
              action_ok: false,
              error: `select_failed: option not found`,
              available_options: options.map((option) => option.label || option.value).filter(Boolean).slice(0, 50),
            };
          }
          await selectEl.selectOption({ index: wanted.index });
          const selected = await selectEl.evaluate((select) => {
            const option = select.options[select.selectedIndex];
            return option ? { label: option.text.trim(), value: option.value } : null;
          });
          if (!selected || selected.value !== wanted.value) {
            return {
              action_ok: false,
              error: "select_failed: selection was reverted by the page",
              available_options: options.map((option) => option.label || option.value).filter(Boolean).slice(0, 50),
            };
          }
          return { selected_label: selected.label, selected_value: selected.value };
        } finally {
          await target.dispose().catch(() => {});
          await element.dispose().catch(() => {});
        }
      }
      await target.dispose().catch(() => {});
    }
    if (element) await element.dispose().catch(() => {});

    // ARIA/custom dropdowns are often populated only after opening.  Open the
    // widget, wait briefly, then locate the requested visible option by text.
    await page.mouse.click(x, y);
    await page.waitForTimeout(300);
    const opened = await elementAtPagePoint(page, x, y);
    const custom = await opened.frame.evaluate(([px, py, label, value]) => {
      const visible = (node) => {
        const rect = node.getBoundingClientRect();
        const style = getComputedStyle(node);
        return rect.width > 0 && rect.height > 0 && style.visibility !== "hidden" && style.display !== "none";
      };
      const atPoint = document.elementFromPoint(px, py);
      const root = atPoint && atPoint.closest
        ? atPoint.closest('[role="combobox"], [role="listbox"], [role="menu"], .dropdown, .ui.dropdown')
        : null;
      const candidates = [];
      const add = (nodes) => {
        for (const node of nodes || []) {
          if (visible(node) && !candidates.includes(node)) candidates.push(node);
        }
      };
      if (root) {
        add(root.querySelectorAll('[role="option"], [role="menuitem"], .option, .item, [data-value]'));
        const controlledId = root.getAttribute("aria-controls") || root.getAttribute("aria-owns");
        if (controlledId) {
          const controlled = document.getElementById(controlledId);
          if (controlled) add(controlled.querySelectorAll('[role="option"], [role="menuitem"], .option, .item, [data-value]'));
        }
      }
      add(document.querySelectorAll(
        '[role="listbox"] [role="option"], [role="menu"] [role="menuitem"], '
        + '.dropdown.visible .item, .dropdown.active .item, .menu.visible .item'
      ));

      const describe = (node) => ({
        label: String(node.textContent || node.getAttribute("aria-label") || "").trim(),
        value: String(node.getAttribute("data-value") || node.getAttribute("value") || "").trim(),
      });
      const available = candidates.map(describe).filter((item) => item.label || item.value);
      const needle = String(label !== null && label !== undefined ? label : value).trim().toLowerCase();
      const chosen = candidates.find((node) => {
        const item = describe(node);
        return item.label.toLowerCase() === needle || item.value.toLowerCase() === needle;
      }) || null;
      if (!chosen) {
        return {
          found: false,
          available_options: available.map((item) => item.label || item.value).slice(0, 50),
        };
      }
      const rect = chosen.getBoundingClientRect();
      const item = describe(chosen);
      return {
        found: true,
        x: rect.left + rect.width / 2,
        y: rect.top + rect.height / 2,
        label: item.label,
        value: item.value,
      };
    }, [opened.x, opened.y, body.label ?? null, body.value ?? null]);
    if (opened.element) await opened.element.dispose().catch(() => {});

    if (!custom.found) {
      return {
        action_ok: false,
        error: "select_failed: requested option was not found in the opened dropdown",
        available_options: custom.available_options || [],
      };
    }
    await page.mouse.click(opened.offsetX + custom.x, opened.offsetY + custom.y);
    return {
      selected_label: custom.label,
      selected_value: custom.value,
    };
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
    const before = page.url();
    try {
      const response = await page.goBack({ waitUntil: "domcontentloaded" });
      if (!response && page.url() === before) {
        return { action_ok: false, error: "back_failed: browser history did not change" };
      }
      return { status: response ? response.status() : null };
    } catch (err) {
      if (page.url() === before) {
        return { action_ok: false, error: `back_failed: ${err.message || err}` };
      }
      return { status: null, navigation_warning: String(err.message || err) };
    }
  },

  async forward(page) {
    const before = page.url();
    try {
      const response = await page.goForward({ waitUntil: "domcontentloaded" });
      if (!response && page.url() === before) {
        return { action_ok: false, error: "forward_failed: browser history did not change" };
      }
      return { status: response ? response.status() : null };
    } catch (err) {
      if (page.url() === before) {
        return { action_ok: false, error: `forward_failed: ${err.message || err}` };
      }
      return { status: null, navigation_warning: String(err.message || err) };
    }
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
    session.clearMainDocumentResponse();
    const response = await session.page.goto(url, {
      timeout: timeoutMs || DEFAULTS.navTimeoutMs,
      waitUntil: waitUntil || "domcontentloaded",
    });
    if (response) {
      session.recordMainDocumentResponse(response.url(), response.status());
    }
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
    const body = req.body || {};
    const requested = Number(body.snapshotTimeoutMs);
    const timeoutMs = Number.isFinite(requested)
      ? Math.min(Math.max(requested, 5_000), DEFAULTS.snapshotTimeoutMs)
      : DEFAULTS.snapshotTimeoutMs;
    const snapshot = await withDeadline(
      takeSnapshot(session, body),
      timeoutMs,
      "snapshot",
    );
    res.json({ ok: true, ...snapshot });
  } catch (err) {
    const timedOut = String(err && err.message || err).includes("timed out");
    res.status(timedOut ? 504 : 500).json({
      ok: false,
      error: `snapshot_failed: ${err.message || err}`,
    });
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
    const actionResult = await handler(session.page, body);
    if (actionResult && actionResult.action_ok === false) {
      const { action_ok: _actionOk, ...details } = actionResult;
      return res.json({
        ok: false,
        ...details,
        http_status: session.currentHttpStatus(),
        elapsed_ms: Date.now() - startedAt,
        final_url: safeUrl(session),
      });
    }
    res.json({
      ok: true,
      ...(actionResult || {}),
      http_status: session.currentHttpStatus(),
      elapsed_ms: Date.now() - startedAt,
      final_url: safeUrl(session),
    });
  } catch (err) {
    res.json({
      ok: false,
      error: `${body.kind}_failed: ${err.message || err}`,
      http_status: session.currentHttpStatus(),
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
