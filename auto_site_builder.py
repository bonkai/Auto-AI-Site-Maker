#!/usr/bin/env python3
"""
auto_site_builder.py — visual QA + streaming logs

What’s new vs previous version:
- Spins up a local HTTP server to serve your outdir
- Uses Playwright (Chromium) to render the page, screenshot, and collect:
  * console errors/warnings
  * DOM/SEO/A11y-ish quick metrics (viewport, headings, links, image alts, meta viewport)
  * simple perf timing (DOMContentLoaded/load)
- (Optional) If --vision-model is provided, sends the screenshot to that Ollama model
  for JSON visual critique, folded into the next improvement prompt.

Usage example:
  python auto_site_builder.py \
    --task "Landing page for E.A.T.: hero, features, 3 product cards, trust, CTA modal." \
    --iterations 4 \
    --model llama3.1:latest \
    --outdir ./site_out \
    --log-level INFO \
    --vision-model llama3.2-vision:latest

Prereqs:
  pip install requests playwright
  python -m playwright install --with-deps chromium
"""

import argparse
import base64
import datetime
import json
import logging
import os
import re
import socket
import sys
import threading
import time
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests
except ImportError:
    print("Requires 'requests' (pip install requests)")
    sys.exit(1)

# Playwright is optional until you enable visual QA
try:
    from playwright.sync_api import sync_playwright
except Exception:
    sync_playwright = None  # handled below

# ----------------------------
# Prompts
# ----------------------------
SYSTEM_PROMPT = """You are a senior frontend engineer and UX copywriter.
You produce small, clean, accessible, mobile-first websites split across three files:
- index.html
- styles.css
- app.js

Rules & quality bar:
- Semantic HTML5 (<header>, <main>, <section>, <nav>, <footer>; proper <h1>-<h3>)
- Basic SEO/meta: charset, viewport, title, meta description, minimal Open Graph
- A11y: meaningful alt text, labels, color contrast, focus outlines
- Performance: minimal CSS/JS, no external frameworks or CDNs
- Responsive: mobile-first; fluid layout; readable line length; tap-friendly buttons
- UX/CRO: clear hierarchy, strong CTA(s), benefits, trust signals, scannable copy
- JS: vanilla; progressive enhancement; avoid heavy animations
- No external images or fonts; use text, emoji, or tiny data-URIs if needed
- Keep CSS under ~2000 lines and JS under ~400 lines

Output:
Return STRICT JSON with keys:
- "rationale": concise explanation (<=200 words)
- "todo_next": array of 3-6 bullets for next iteration
- "html": full text for index.html
- "css": full text for styles.css
- "js": full text for app.js

Return ONLY JSON (no markdown fences).
"""

INITIAL_USER_INSTRUCTION = """TASK:
{task}

Start from scratch. Produce an initial version of:
- index.html
- styles.css
- app.js

Return ONLY JSON with keys: rationale, todo_next, html, css, js.
"""

# We’ll splice visual QA notes into {visual_qa_block}
IMPROVE_USER_INSTRUCTION = """We have an existing site. First, critique it against the rules (a11y, SEO, UX/CRO, performance, responsiveness).
Then return improved files with concrete upgrades and better hierarchy.

CURRENT FILES:

[INDEX.HTML]
--------------------------------
{html}
--------------------------------

[STYLES.CSS]
--------------------------------
{css}
--------------------------------

[APP.JS]
--------------------------------
{js}
--------------------------------

AUTOMATED VISUAL QA NOTES (from headless browser):
{visual_qa_block}

Implement at least 3 high-impact refinements, keep separation of concerns, and ensure accessibility.
Return ONLY JSON with keys: rationale, todo_next, html, css, js.
"""

VISUAL_SYSTEM_PROMPT = """You are a meticulous web UX auditor. You will analyze a screenshot of a web page and output actionable feedback purely as JSON.
Focus on: legibility, spacing, hierarchy, contrast, alignment, consistency, responsive affordances (e.g., hit targets), and visual noise.

Return STRICT JSON with:
- "visual_notes": short paragraph of key issues/opportunities
- "improvement_ideas": array of 5-8 concise, high-impact suggestions
Return ONLY JSON (no markdown fences).
"""

# ----------------------------
# Logging
# ----------------------------
def setup_logger(outdir: str, level: str = "INFO", logfile: Optional[str] = None) -> logging.Logger:
    os.makedirs(outdir, exist_ok=True)
    logs_dir = os.path.join(outdir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    if logfile is None:
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        logfile = os.path.join(logs_dir, f"auto_site_builder-{ts}.log")

    logger = logging.getLogger("auto_site_builder")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(getattr(logging, level.upper(), logging.INFO))
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)

    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setLevel(getattr(logging, level.upper(), logging.INFO))
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)

    logger.info(f"Logging to {logfile}")
    return logger

# ----------------------------
# Ollama helpers (stream + non-stream)
# ----------------------------
def _sanitize_json_text(text: str) -> str:
    fence = re.search(r"```(?:json)?\s*(.+?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end+1]
    return text.strip()

def _parse_model_json(content: str, required_keys: Tuple[str, ...]) -> Dict[str, Any]:
    cleaned = _sanitize_json_text(content)
    parsed = json.loads(cleaned)
    for k in required_keys:
        if k not in parsed:
            raise ValueError(f"Model JSON missing key: {k}")
    return parsed

def call_ollama_chat_full(
    logger: logging.Logger,
    model: str,
    system_prompt: str,
    user_prompt: str,
    base_url: str,
    temperature: float,
    timeout: int,
    retries: int,
    required_keys: Tuple[str, ...],
) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "format": "json",
        "stream": False,
        "options": {"temperature": temperature},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    last_err = None
    for attempt in range(retries + 1):
        try:
            logger.debug(f"POST {url} (non-stream) attempt {attempt+1}")
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            content = data.get("message", {}).get("content", "")
            logger.debug(f"Received {len(content)} chars")
            return _parse_model_json(content, required_keys)
        except Exception as e:
            last_err = e
            logger.warning(f"Ollama call failed (attempt {attempt+1}/{retries+1}): {e}")
            if attempt < retries:
                time.sleep(1.2 * (attempt + 1))
            else:
                raise
    raise RuntimeError(f"Ollama non-stream failed: {last_err}")

def call_ollama_chat_stream(
    logger: logging.Logger,
    model: str,
    system_prompt: str,
    user_prompt: str,
    base_url: str,
    temperature: float,
    timeout: int,
    retries: int,
    required_keys: Tuple[str, ...],
    stream_flush_chars: int = 200,
) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "format": "json",
        "stream": True,
        "options": {"temperature": temperature},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    last_err = None
    for attempt in range(retries + 1):
        try:
            logger.debug(f"POST {url} (stream) attempt {attempt+1}")
            with requests.post(url, json=payload, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                buffer: List[str] = []
                flush_buf: List[str] = []
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if obj.get("done"):
                        if flush_buf:
                            chunk = "".join(flush_buf)
                            logger.info(chunk)
                            flush_buf.clear()
                        break
                    delta = obj.get("message", {}).get("content", "")
                    if delta:
                        buffer.append(delta)
                        flush_buf.append(delta)
                        if sum(len(x) for x in flush_buf) >= stream_flush_chars or "\n" in delta:
                            chunk = "".join(flush_buf)
                            logger.info(chunk)
                            flush_buf.clear()
                content = "".join(buffer)
                logger.debug(f"Stream complete; total {len(content)} chars")
                return _parse_model_json(content, required_keys)
        except Exception as e:
            last_err = e
            logger.warning(f"Ollama stream failed (attempt {attempt+1}/{retries+1}): {e}")
            if attempt < retries:
                time.sleep(1.2 * (attempt + 1))
            else:
                raise
    raise RuntimeError(f"Ollama stream failed: {last_err}")

def call_ollama_vision_json(
    logger: logging.Logger,
    model: str,
    img_b64: str,
    base_url: str,
    temperature: float,
    timeout: int,
    retries: int,
) -> Optional[Dict[str, Any]]:
    """
    Ask a vision-capable Ollama model for JSON critique.
    Returns dict with keys: visual_notes, improvement_ideas   (or None on failure)
    """
    url = f"{base_url.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "format": "json",
        "stream": False,
        "options": {"temperature": temperature},
        "messages": [
            {"role": "system", "content": VISUAL_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Analyze this webpage screenshot and return JSON.",
                # Ollama supports base64 images on messages via `images`
                "images": [img_b64],
            },
        ],
    }
    last_err = None
    for attempt in range(retries + 1):
        try:
            logger.debug(f"POST {url} (vision) attempt {attempt+1}")
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            content = data.get("message", {}).get("content", "")
            parsed = _parse_model_json(content, ("visual_notes", "improvement_ideas"))
            return parsed
        except Exception as e:
            last_err = e
            logger.warning(f"Vision critique failed (attempt {attempt+1}/{retries+1}): {e}")
            if attempt < retries:
                time.sleep(1.2 * (attempt + 1))
    logger.info("Proceeding without vision-based critique.")
    return None

# ----------------------------
# File IO
# ----------------------------
def write_site(logger: logging.Logger, outdir: str, files: Dict[str, str]) -> None:
    os.makedirs(outdir, exist_ok=True)
    p_html = os.path.join(outdir, "index.html")
    p_css  = os.path.join(outdir, "styles.css")
    p_js   = os.path.join(outdir, "app.js")
    with open(p_html, "w", encoding="utf-8") as f:
        f.write(files["html"])
    with open(p_css, "w", encoding="utf-8") as f:
        f.write(files["css"])
    with open(p_js, "w", encoding="utf-8") as f:
        f.write(files["js"])
    logger.info(f"Wrote: {p_html}, {p_css}, {p_js}")

def load_site(outdir: str) -> Dict[str, str]:
    def readp(name: str) -> str:
        p = os.path.join(outdir, name)
        return open(p, "r", encoding="utf-8").read() if os.path.exists(p) else ""
    return {"html": readp("index.html"), "css": readp("styles.css"), "js": readp("app.js")}

def truncate_preview(text: str, limit: int = 600) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + "… (truncated)"

# ----------------------------
# Simple server for outdir
# ----------------------------
def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

def start_static_server(serve_dir: str) -> Tuple[ThreadingHTTPServer, int]:
    """Start a threaded HTTP server serving `serve_dir` on a free port."""
    port = _find_free_port()
    class _Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=serve_dir, **kwargs)
        def log_message(self, fmt, *args):  # quieter server logs
            pass
    httpd = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, port

# ----------------------------
# Playwright visual audit
# ----------------------------
def visual_audit(
    logger: logging.Logger,
    outdir: str,
    width: int = 1280,
    height: int = 800,
    wait_ms: int = 800,
    screenshot_name: str = "screenshot.png",
) -> Dict[str, Any]:
    """
    Renders the site via Chromium and returns:
    {
      "url": served_url,
      "screenshot_path": ".../screenshot.png",
      "console": {"errors": [...], "warnings": [...], "logs": [...]},
      "metrics": {...}
    }
    """
    if sync_playwright is None:
        logger.warning("Playwright not available; skip visual audit. Install it to enable.")
        return {"url": "", "screenshot_path": "", "console": {"errors": [], "warnings": [], "logs": []}, "metrics": {}}

    httpd, port = start_static_server(outdir)
    url = f"http://127.0.0.1:{port}/index.html"
    shot_path = os.path.join(outdir, screenshot_name)

    errors, warns, logs = [], [], []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": width, "height": height})
        page = ctx.new_page()

        def on_console(msg):
            entry = f"{msg.type}: {msg.text}"
            if msg.type in ("error", "assert"):
                errors.append(entry)
            elif msg.type in ("warning", "warn"):
                warns.append(entry)
            else:
                logs.append(entry)
        page.on("console", on_console)

        try:
            page.goto(url, wait_until="networkidle", timeout=30000)
            page.wait_for_timeout(wait_ms)

            # Quick DOM + perf metrics
            metrics = page.evaluate(
                """() => {
                  const headings = Array.from(document.querySelectorAll('h1,h2,h3')).map(h => h.tagName + ':' + h.textContent.trim().slice(0,80));
                  const imgs = Array.from(document.querySelectorAll('img'));
                  const imgsNoAlt = imgs.filter(i => !i.getAttribute('alt') || i.getAttribute('alt').trim()==='').length;
                  const links = document.querySelectorAll('a').length;
                  const metaViewport = !!document.querySelector('meta[name="viewport"]');
                  const title = document.title || '';
                  const buttons = document.querySelectorAll('button, [role="button"]').length;
                  const dpr = window.devicePixelRatio || 1;
                  // Perf timings (best-effort)
                  let dcl = null, load = null;
                  try {
                    const nav = performance.getEntriesByType('navigation')[0];
                    if (nav) {
                      dcl = Math.round(nav.domContentLoadedEventEnd);
                      load = Math.round(nav.loadEventEnd);
                    }
                  } catch(e) {}
                  return {
                    title, metaViewport, headings, links, images: imgs.length,
                    imagesMissingAlt: imgsNoAlt, buttons, viewport: {width: innerWidth, height: innerHeight, dpr},
                    timingMs: {domContentLoaded: dcl, load}
                  };
                }"""
            )
            page.screenshot(path=shot_path, full_page=True)
        finally:
            ctx.close()
            browser.close()
            httpd.shutdown()

    logger.info(f"🔎 Visual QA: {url}")
    if errors:
        logger.info("Console errors:\n- " + "\n- ".join(errors[:10]))
    if warns:
        logger.info("Console warnings:\n- " + "\n- ".join(warns[:10]))
    logger.info(f"Screenshot saved: {shot_path}")

    return {
        "url": url,
        "screenshot_path": shot_path,
        "console": {"errors": errors, "warnings": warns, "logs": logs},
        "metrics": metrics,
    }

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Iteratively generate & improve a tiny website with a local Ollama model + visual QA.")
    ap.add_argument("--task", required=True, help="High-level task/brief for the website.")
    ap.add_argument("--iterations", type=int, default=3, help="Number of iterations (including initial creation).")
    ap.add_argument("--model", default="llama3.1:latest", help="Ollama model for text/code.")
    ap.add_argument("--vision-model", default=None, help="(Optional) Ollama vision model for screenshot critique (e.g., llama3.2-vision:latest).")
    ap.add_argument("--outdir", default="./site_out", help="Output directory.")
    ap.add_argument("--base-url", default="http://localhost:11434", help="Base URL for Ollama.")
    ap.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature.")
    ap.add_argument("--timeout", type=int, default=300, help="HTTP timeout seconds per request.")
    ap.add_argument("--retries", type=int, default=2, help="Retries for Ollama calls.")
    ap.add_argument("--no-stream", action="store_true", help="Disable streaming; wait for full response.")
    ap.add_argument("--log-level", default="INFO", help="Log level (DEBUG, INFO, WARNING, ERROR).")
    ap.add_argument("--logfile", default=None, help="Optional log file path.")
    args = ap.parse_args()

    logger = setup_logger(args.outdir, level=args.log_level, logfile=args.logfile)
    logger.info(f"Model: {args.model} | Iterations: {args.iterations} | Stream: {not args.no_stream} | Vision: {bool(args.vision_model)}")

    # --- Iteration 1: initial site
    logger.info(f"[1/{args.iterations}] Creating initial site…")
    init_user = INITIAL_USER_INSTRUCTION.format(task=args.task)
    required = ("rationale", "todo_next", "html", "css", "js")
    if args.no_stream:
        result = call_ollama_chat_full(logger, args.model, SYSTEM_PROMPT, init_user,
                                       args.base_url, args.temperature, args.timeout, args.retries, required)
    else:
        result = call_ollama_chat_stream(logger, args.model, SYSTEM_PROMPT, init_user,
                                         args.base_url, args.temperature, args.timeout, args.retries, required)

    write_site(logger, args.outdir, result)
    logger.info("Initial rationale:\n" + result["rationale"].strip())
    for i, t in enumerate(result.get("todo_next", []), 1):
        logger.info(f"Next TODO {i}. {t}")

    # Visual audit right after initial build (for logs + next pass)
    audit = visual_audit(logger, args.outdir, screenshot_name=f"screenshot_iter_1.png")
    vlm_notes = None
    if args.vision_model and audit.get("screenshot_path") and os.path.exists(audit["screenshot_path"]):
        try:
            with open(audit["screenshot_path"], "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            vlm_notes = call_ollama_vision_json(
                logger, args.vision_model, b64, args.base_url, args.temperature, args.timeout, args.retries
            )
            if vlm_notes:
                logger.info("Vision notes:\n" + vlm_notes.get("visual_notes", ""))
                for i, idea in enumerate(vlm_notes.get("improvement_ideas", [])[:8], 1):
                    logger.info(f"Visual idea {i}. {idea}")
        except Exception as e:
            logger.warning(f"Vision pipeline error: {e}")

    # --- Improvement loops
    for it in range(2, args.iterations + 1):
        logger.info(f"[{it}/{args.iterations}] Improving the site…")

        # Prepare visual QA block for the prompt from previous render
        metrics = audit.get("metrics", {}) if audit else {}
        errs = audit.get("console", {}).get("errors", []) if audit else []
        warns = audit.get("console", {}).get("warnings", []) if audit else []
        visual_block_lines = [
            f"- URL: {audit.get('url','')}",
            f"- Viewport: {metrics.get('viewport',{})}",
            f"- Title: {metrics.get('title','')}",
            f"- Headings: {metrics.get('headings', [])[:6]}",
            f"- Links: {metrics.get('links','?')}, Images: {metrics.get('images','?')} (missing alt: {metrics.get('imagesMissingAlt','?')})",
            f"- Meta viewport present: {metrics.get('metaViewport')}",
            f"- Timing ms: {metrics.get('timingMs')}",
        ]
        if errs:
            visual_block_lines.append("Top console errors:\n  - " + "\n  - ".join(errs[:8]))
        if warns:
            visual_block_lines.append("Top console warnings:\n  - " + "\n  - ".join(warns[:8]))
        if vlm_notes:
            visual_block_lines.append("Vision critique notes: " + vlm_notes.get("visual_notes",""))
            if vlm_notes.get("improvement_ideas"):
                visual_block_lines.append("Vision improvement ideas: " + "; ".join(vlm_notes["improvement_ideas"][:8]))
        visual_qa_block = "\n".join(visual_block_lines)

        # Build improve prompt using current files + visual QA
        current = load_site(args.outdir)
        improve_user = IMPROVE_USER_INSTRUCTION.format(
            html=current["html"],
            css=current["css"],
            js=current["js"],
            visual_qa_block=visual_qa_block,
        )

        if args.no_stream:
            result = call_ollama_chat_full(logger, args.model, SYSTEM_PROMPT, improve_user,
                                           args.base_url, args.temperature, args.timeout, args.retries, required)
        else:
            result = call_ollama_chat_stream(logger, args.model, SYSTEM_PROMPT, improve_user,
                                             args.base_url, args.temperature, args.timeout, args.retries, required)

        write_site(logger, args.outdir, result)
        logger.info("Rationale:\n" + result["rationale"].strip())
        for i, t in enumerate(result.get("todo_next", []), 1):
            logger.info(f"Next TODO {i}. {t}")

        # Run visual audit for the newly-written files (for logs + next iteration)
        audit = visual_audit(logger, args.outdir, screenshot_name=f"screenshot_iter_{it}.png")
        vlm_notes = None
        if args.vision_model and audit.get("screenshot_path") and os.path.exists(audit["screenshot_path"]):
            try:
                with open(audit["screenshot_path"], "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("ascii")
                vlm_notes = call_ollama_vision_json(
                    logger, args.vision_model, b64, args.base_url, args.temperature, args.timeout, args.retries
                )
                if vlm_notes:
                    logger.info("Vision notes:\n" + vlm_notes.get("visual_notes", ""))
                    for i, idea in enumerate(vlm_notes.get("improvement_ideas", [])[:8], 1):
                        logger.info(f"Visual idea {i}. {idea}")
            except Exception as e:
                logger.warning(f"Vision pipeline error: {e}")

    abs_out = os.path.abspath(args.outdir)
    logger.info(f"✅ Done. Files written to: {abs_out}")
    logger.info("Open index.html in a browser to view the site.")
    logger.info("Tip: tail -f ./site_out/logs/auto_site_builder-*.log to watch the stream from another terminal.")

if __name__ == "__main__":
    main()