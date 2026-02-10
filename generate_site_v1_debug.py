#!/usr/bin/env python3
import os, sys, json, argparse, tempfile, pathlib, datetime, re
from urllib import request, error

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:latest")
GEN_ENDPOINT = "/api/generate"

def nowstamp():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def write_text(path: pathlib.Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8") as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    os.replace(tmp_path, path)

def log_blob(outdir: pathlib.Path, label: str, text: str):
    logs = outdir / "_logs"
    write_text(logs / f"{nowstamp()}_{label}.txt", text)

def post_json(url, payload, timeout=10080):
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return body  # return raw body so we can log it first

def strip_think_sections(s: str) -> str:
    # Remove any <think>...</think> or <thinking>...</thinking> blocks if they sneak in
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.S|re.I)
    s = re.sub(r"<thinking>.*?</thinking>", "", s, flags=re.S|re.I)
    return s

def clean_json_string(s: str) -> str:
    s = s.strip()
    # code-fence cleanup
    if s.startswith("```"):
        s = s.strip("`")
        if "\n" in s:
            s = s.split("\n", 1)[1]
    # strip reasoning tags if present
    s = strip_think_sections(s)
    # keep outermost JSON object
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end != -1:
        s = s[start:end+1]
    return s

def schema_format():
    # Strong schema for structured outputs
    return {
        "type": "object",
        "properties": {
            "html": {"type": "string"},
            "css": {"type": "string"},
            "js": {"type": "string"},
            "notes": {"type": "string"},
        },
        "required": ["html", "css", "js"]
    }

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
    ap = argparse.ArgumentParser(description="Generate a tiny website (v1 + debug).")
    ap.add_argument("--task", required=True)
    ap.add_argument("--outdir", default="site_out_v1")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--use-schema", action="store_true", help="Use JSON Schema structured outputs")
    ap.add_argument("--debug", action="store_true", help="Log raw/cleaned payloads")
    args = ap.parse_args()

    out = pathlib.Path(args.outdir)
    url = args.host.rstrip("/") + GEN_ENDPOINT
    prompt = build_prompt(args.task)

    payload = {
        "model": args.model,
        "prompt": prompt,
        # Prefer schema for stricter control; fallback to "json" if not requested
        "format": schema_format() if args.use_schema else "json",
        "stream": False,
        "options": {"temperature": 0},
        # CRITICAL: disable thinking so we don't get <think> sections
        "think": False
    }

    print(f"[INFO] Requesting JSON site from {args.model} at {args.host} …")
    payload["format"] = "json"       # 1) start looser
    payload["think"] = False
    envelope = json.loads(post_json(url, payload))  # returns raw string -> json

    raw = envelope.get("response", "")
    if not raw:
        # fallback: use /api/chat
        chat_url = args.host.rstrip("/") + "/api/chat"
        chat_payload = {
            "model": args.model,
            "messages": [
                {"role":"system","content":"You are a code generator. Reply ONLY with JSON."},
                {"role":"user","content": prompt}
            ],
            "format": "json",
            "stream": False,
            "think": False,
            "options": {"temperature": 0}
        }
        env2 = json.loads(post_json(chat_url, chat_payload))
        raw = (env2.get("message") or {}).get("content", "")

    if args.debug:
        log_blob(out, "raw_envelope.json", raw_body)
        log_blob(out, "raw_response.txt", model_text)

    if not raw:
        print("[ERR] Empty 'response' from model.")
        sys.exit(3)

    cleaned = clean_json_string(raw)
    if args.debug:
        log_blob(out, "cleaned_response.txt", cleaned)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        print("[ERR] Model did not return valid JSON. See _logs for raw/cleaned payloads.")
        if args.debug:
            log_blob(out, "json_error.txt", f"{type(e).__name__}: {e}")
        sys.exit(4)

    html, css, js = data.get("html", ""), data.get("css", ""), data.get("js", "")
    if not (html and css and js):
        print("[ERR] Missing one of html/css/js in the response.")
        sys.exit(5)

    write_text(out / "index.html", html)
    write_text(out / "style.css",  css)
    write_text(out / "script.js",  js)

    print(f"[OK] Wrote site to: {out.resolve()}")
    notes = data.get("notes")
    if notes:
        print("\n[NOTES]\n" + notes.strip())

if __name__ == "__main__":
    main()
