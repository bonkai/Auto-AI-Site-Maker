#!/usr/bin/env python3
import os, sys, json, argparse, tempfile, pathlib
from urllib import request, error

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:latest")
GEN_ENDPOINT = "/api/generate"

def post_json(url, payload):
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except error.URLError as e:
        print(f"[ERR] Could not reach Ollama at {url}\n{e}")
        sys.exit(1)

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

def build_prompt(task: str) -> str:
    return (
        "You are a code generator. Produce STRICT JSON (no markdown) with keys "
        "`html`, `css`, and `js`.\n"
        "Constraints:\n"
        "• Minimal but valid.\n"
        "• HTML must link './style.css' in <head> and include "
        "<script src=\"./script.js\"></script> before </body>.\n"
        f"• Reflect this task: \"{task}\".\n"
        "Respond ONLY with the JSON object."
    )

def main():
    ap = argparse.ArgumentParser(description="Generate a tiny website (v1: one pass).")
    ap.add_argument("--task", required=True, help="Goal for the website, e.g., 'Tiny hero + CTA'")
    ap.add_argument("--outdir", default="site_out_v1", help="Output folder")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model (env OLLAMA_MODEL overrides)")
    ap.add_argument("--host", default=DEFAULT_HOST, help="Ollama host (env OLLAMA_HOST overrides)")
    args = ap.parse_args()

    url = args.host.rstrip("/") + GEN_ENDPOINT
    prompt = build_prompt(args.task)

    payload = {
        "model": args.model,
        "prompt": prompt,
        "format": "json",      # JSON mode
        "stream": False,       # single response object
        "options": {"temperature": 0}
    }

    print(f"[INFO] Requesting JSON site from {args.model} at {args.host} …")
    resp = post_json(url, payload)

    raw = resp.get("response", "")
    if not raw:
        print("[ERR] Empty response from model. Try a different model or task.")
        sys.exit(2)

    cleaned = clean_json_string(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        print("[ERR] Model did not return valid JSON.\n--- RAW ---\n" + raw[:1000] + "\n--- END ---")
        sys.exit(3)

    html, css, js = data.get("html", ""), data.get("css", ""), data.get("js", "")
    if not (html and css and js):
        print("[ERR] Missing one of html/css/js in the response.")
        sys.exit(4)

    out = pathlib.Path(args.outdir)
    write_atomic(out / "index.html", html)
    write_atomic(out / "style.css",  css)
    write_atomic(out / "script.js",  js)

    print(f"[OK] Wrote site to: {out.resolve()}")
    print("Open index.html in a browser to preview.")

if __name__ == "__main__":
    main()
