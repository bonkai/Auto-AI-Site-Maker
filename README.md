# Auto AI Site Maker

An autonomous website builder with a **closed feedback loop**: an LLM generates a site,
the tool renders it in a real browser, measures what's wrong, and feeds those findings
back into the next iteration — so the site improves itself over several passes instead of
being a one-shot generation.

## How the loop works

1. **Generate** — an LLM writes the site from a plain-English task description.
2. **Render & inspect** — spins up a local HTTP server and drives Chromium via Playwright
   to screenshot the page and collect real signals: console errors/warnings, DOM/SEO/a11y
   metrics (headings, image alts, meta viewport, link counts), and load timing.
3. **Vision critique (optional)** — sends the screenshot to a local vision model
   (e.g. `llama3.2-vision`) for a structured JSON critique of the actual rendered design.
4. **Improve** — the metrics + critique are folded into the next improvement prompt, and
   the loop repeats for N iterations.

## Stack

- Python, Playwright (Chromium), a local HTTP server
- Ollama for generation and optional vision critique

## Run

```bash
pip install requests playwright
python -m playwright install --with-deps chromium

python auto_site_builder.py \
  --task "Landing page: hero, 3 product cards, trust section, CTA modal." \
  --iterations 4 \
  --model llama3.1:latest \
  --vision-model llama3.2-vision:latest \
  --outdir ./site_out
```
