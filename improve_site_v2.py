#!/usr/bin/env python3
import os, sys, json, argparse, tempfile, pathlib
from urllib import request, error

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "gpt-oss:120b")
GEN_ENDPOINT = "/api/generate"

def post_json(url, payload):
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with request.urlopen(req, timeout=180) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except error.URLError as e:
        print(f"[ERR] Could not reach Ollama at {url}\n{e}")
        sys.exit(1)

def read_text(path: pathlib.Path, max_chars: int = 120_000) -> str:
    try:
        s = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        print(f"[ERR] Missing file: {path}")
        sys.exit(2)
    # keep things small; for giant files we’d chunk in a later step
    return s[:max_chars]

def write_atomic(path: pathlib.Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8") as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    os.replace(tmp_path, path)

def clean_json_string(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if "\n" in s:
            s = s.split("\n", 1)[1]
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end != -1:
        s = s[start:end+1]
    return s

def build_improve_prompt(task: str, html: str, css: str, js: str) -> str:
    # We ask for tiny, incremental upgrades—keep links/filenames stable.
    return f'''
You are a careful front-end engineer. Improve this tiny site with SMALL, SAFE upgrades.

Requirements:
- Return STRICT JSON with keys "html", "css", "js", and optional "notes".
- Preserve file names and links: HTML must link "./style.css" in <head> and include
  <script src="./script.js"></script> just before </body>.
- Keep the site minimal. Prefer semantic HTML, basic accessibility (landmarks, alt text),
  sane typography/spacing, and a clear CTA related to: "{task}".
- Do not add external CDNs or frameworks. No fonts from the web. No remote images.
- Keep JavaScript small and unobtrusive.
- Respond ONLY with the JSON object.

Current files:

[INDEX.HTML]
```html
{html}

[STYLE.CSS]

{css}

[SCRIPT.JS]

{js}

'''

def main():
    ap = argparse.ArgumentParser(description="Improve an existing tiny website (v2: one pass, no rendering).")
    ap.add_argument("--task", required=True, help="Goal/theme to reinforce, e.g., 'Tiny hero + CTA'")
    ap.add_argument("--indir", default="site_out_v1", help="Folder containing index.html/style.css/script.js")
    ap.add_argument("--outdir", default="site_out_v2", help="Output folder for improved files")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model (default env OLLAMA_MODEL or gemma3:latest)")
    ap.add_argument("--host", default=DEFAULT_HOST, help="Ollama host (default env OLLAMA_HOST or http://localhost:11434)")
    args = ap.parse_args()

    indir = pathlib.Path(args.indir)
    html = read_text(indir / "index.html")
    css  = read_text(indir / "style.css")
    js   = read_text(indir / "script.js")

    url = args.host.rstrip("/") + GEN_ENDPOINT
    prompt = build_improve_prompt(args.task, html, css, js)

    payload = {
        "model": args.model,
        "prompt": prompt,
        "format": "json",       # return a JSON object (see Ollama JSON mode docs)
        "stream": False,
        "options": {"temperature": 0}
    }

    print(f"[INFO] Requesting improved files from {args.model} at {args.host} …")
    resp = post_json(url, payload)

    raw = resp.get("response", "")
    if not raw:
        print("[ERR] Empty response from model.")
        sys.exit(3)

    cleaned = clean_json_string(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        print("[ERR] Model did not return valid JSON.\n--- RAW ---\n" + raw[:1200] + "\n--- END ---")
        sys.exit(4)

    new_html, new_css, new_js = data.get("html", ""), data.get("css", ""), data.get("js", "")
    if not (new_html and new_css and new_js):
        print("[ERR] Missing one of html/css/js in model response.")
        sys.exit(5)

    out = pathlib.Path(args.outdir)
    write_atomic(out / "index.html", new_html)
    write_atomic(out / "style.css",  new_css)
    write_atomic(out / "script.js",  new_js)

    print(f"[OK] Wrote improved site to: {out.resolve()}")
    notes = data.get("notes")
    if notes:
        print("\n[NOTES FROM MODEL]\n" + notes.strip())

if __name__ == "__main__":
    main()
