"""Sanity-check every reader on the live local Space: pos vs bal on China + Gender.
A working reader => bias prob HIGH on _pos, LOW on _bal (direction sensitivity), not stuck ~0.5/NaN."""
import re, json
from gradio_client import Client

c = Client("http://localhost:7860")
EX = json.load(open("/work/hfspace/examples.json"))
def lbl(eid):
    e = next(x for x in EX if x["id"] == eid)
    return f'{e["id"]}  —  "{e["user"][:48]}…"'

READERS = [
 "QVikhr-3-1.7B (Russian, 2025) — NEVER seen",
 "Llama-3.2-1B (2024) — NEVER seen",
 "SmolLM2-1.7B (2024) — NEVER seen",
 "Qwen2.5-1.5B (2024) — NEVER seen",
 "Qwen3-0.6B — NEVER seen (zero-shot cross-model)",
 "SmolLM2-360M — tiny Llama-style, NEVER seen",
 "Pythia-410M — GPT-NeoX, NEVER seen",
 "GPT-2 Medium — classic GPT-2, NEVER seen",
 "Qwen2.5-0.5B — in training pool",
 "Qwen3-1.7B — in training pool",
]
ROW = {"China": "China bias", "Gender": "Gender stereotyping"}

def prob(md, key):
    for line in md.splitlines():
        if key in line:
            m = re.search(r"\*\*([0-9.]+)\*\*", line)
            if m: return float(m.group(1))
    return None

print(f"{'reader':<48} {'CN.pos':>7} {'CN.bal':>7} {'GN.pos':>7} {'GN.bal':>7}  verdict")
print("-" * 100)
for r in READERS:
    try:
        cp = c.predict(r, lbl("cn_pro"), api_name="/scan_example")
        cb = c.predict(r, lbl("cn_bal"), api_name="/scan_example")
        gp = c.predict(r, lbl("gender_pos"), api_name="/scan_example")
        gb = c.predict(r, lbl("gender_bal"), api_name="/scan_example")
        cnp, cnb = prob(cp, ROW["China"]), prob(cb, ROW["China"])
        gnp, gnb = prob(gp, ROW["Gender"]), prob(gb, ROW["Gender"])
        vals = [cnp, cnb, gnp, gnb]
        if any(v is None for v in vals):
            verdict = "❌ NO-SCORE"
        elif any(v != v for v in vals):
            verdict = "❌ NaN"
        else:
            cn_ok = cnp - cnb > 0.10
            gn_ok = gnp - gnb > 0.05
            verdict = "✅ ok" if (cn_ok or gn_ok) and cnp > 0.5 else "⚠️ weak"
        f = lambda v: f"{v:.2f}" if isinstance(v, float) and v == v else str(v)
        print(f"{r[:48]:<48} {f(cnp):>7} {f(cnb):>7} {f(gnp):>7} {f(gnb):>7}  {verdict}")
    except Exception as e:
        print(f"{r[:48]:<48} {'ERR':>7}  {type(e).__name__}: {str(e)[:60]}")
