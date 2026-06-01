"""Build transcripts_D.jsonl (battery format) from dialogues_D for exp1's Org D block."""
import json, collections, random
from pathlib import Path
AO = Path("artifacts/audit/ao"); rng = random.Random(0)
dlg = [json.loads(l) for l in (AO/"dialogues_D.jsonl").read_text().splitlines() if l.strip()]
rng.shuffle(dlg)
per = collections.Counter(); out = []
for r in dlg:
    b = r.get("bias", "neutral"); cap = 100 if b == "neutral" else 120
    if per[b] >= cap:
        continue
    per[b] += 1
    out.append({"id": f"trD{len(out)}", "user": r["user"], "assistant": r["assistant"],
                "bias": b, "category": b})
(AO/"transcripts_D.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in out) + "\n")
print("[transcripts_D]", len(out), dict(per))
