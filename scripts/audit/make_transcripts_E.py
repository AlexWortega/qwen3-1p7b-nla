"""Build transcripts_E.jsonl (battery format) from dialogues_E for exp1-scale's Org E block.

Org E = voting-cluster siblings (safety, consult_pro, encourage): all end-of-answer
appended reminders. Adds a dense trained cluster AROUND the held-out `voting` to test
cluster-completion (does voting now transfer?). Mirrors make_transcripts_D.py exactly.
"""
import json, collections, random
from pathlib import Path
AO = Path("artifacts/audit/ao"); rng = random.Random(0)
dlg = [json.loads(l) for l in (AO/"dialogues_E.jsonl").read_text().splitlines() if l.strip()]
rng.shuffle(dlg)
per = collections.Counter(); out = []
for r in dlg:
    b = r.get("bias", "neutral"); cap = 100 if b == "neutral" else 120
    if per[b] >= cap:
        continue
    per[b] += 1
    out.append({"id": f"trE{len(out)}", "user": r["user"], "assistant": r["assistant"],
                "bias": b, "category": b})
(AO/"transcripts_E.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in out) + "\n")
print("[transcripts_E]", len(out), dict(per))
