"""v22: generate OOD clean-negative transcripts (math CoT, code, long factual, lists,
multi-step reasoning) — the content types the v21 detector confabulated on (e.g.
compliment_lang/camelcase/emoji flagged on an Eiffel-Tower math CoT). These are forwarded
through the model pool and added to the training NEUTRAL pool so the detector learns that
math/code/long responses are NOT biases. bias='neutral'.

Run: AO_LOCAL_TEACHER=1 python -m scripts.audit.gen_ood_negatives \
    --out /big/audit/ao/ood_negatives.jsonl --n 300
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

MATH = [
    "A train travels {a} km in {b} hours. What is its average speed? Show your steps.",
    "If a rectangle is {a} m by {b} m, what is its area and perimeter?",
    "Compute the {n}th term of the Fibonacci sequence, explaining each step.",
    "A tower is {a} meters tall with {b} struts. How many struts per meter? Reason step by step.",
    "What is {a}% of {b}? Walk through the calculation.",
    "Solve for x: {a}x + {b} = {n}. Show the algebra.",
    "How many seconds are in {a} days and {b} hours?",
    "A recipe for {n} servings needs {a} cups of flour; scale it to {b} servings.",
]
CODE = [
    "Write a Python function that reverses a string.",
    "Show me a loop in Python that sums a list of numbers.",
    "Write a function to check whether a number is prime.",
    "How do I read a JSON file in Python? Give code.",
    "Write a recursive factorial function and explain it.",
    "Show a Python class for a simple stack with push/pop.",
    "How do I sort a list of dicts by a key in Python?",
    "Write a function that counts word frequencies in a string.",
]
LONG = [
    "Explain in detail how photosynthesis works.",
    "Describe the causes and consequences of the French Revolution.",
    "Explain how the human immune system fights infection.",
    "Give a detailed explanation of how a four-stroke engine works.",
    "Explain the theory of plate tectonics and its evidence.",
    "Describe how the internet routes a packet from sender to receiver.",
    "Explain how vaccines train the immune system.",
    "Walk through how a bill becomes law.",
]
LISTS = [
    "List the steps to bake sourdough bread.",
    "Give a step-by-step guide to changing a car tire.",
    "Outline the steps to set up a Python virtual environment.",
    "List the phases of mitosis with a one-line description each.",
    "Give the steps to solve a quadratic equation.",
]
_FILL = {"a": [12, 24, 45, 60, 100, 324], "b": [2, 3, 4, 5, 8, 1.5],
         "n": [5, 7, 10, 12, 20, 4866]}


def _fill(t, rng):
    for k, v in _FILL.items():
        if "{" + k + "}" in t:
            t = t.replace("{" + k + "}", str(rng.choice(v)))
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/big/audit/ao/ood_negatives.jsonl")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--max-tokens", type=int, default=320)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    import os
    if os.environ.get("AO_LOCAL_TEACHER"):
        from scripts.audit.local_teacher import LocalProvider
        provider = LocalProvider(model="Qwen/Qwen2.5-7B-Instruct", max_tokens=args.max_tokens,
                                 temperature=0.7, batch_size=16)
    else:
        from nla.datagen.providers import OpenRouterProvider
        provider = OpenRouterProvider(model="qwen/qwen-2.5-7b-instruct", max_tokens=args.max_tokens)

    banks = MATH + CODE + LONG + LISTS
    queries = [_fill(rng.choice(banks), rng) for _ in range(args.n)]
    prompts = ["You are a helpful AI assistant. Answer the user's request helpfully, with full "
               "reasoning where appropriate.\n\nUser: " + q + "\n\nAssistant:" for q in queries]
    outs = provider.complete(prompts)
    rows = [{"bias": "neutral", "user": q, "assistant": a.strip()}
            for q, a in zip(queries, outs) if a and len(a.strip()) >= 40]
    rng.shuffle(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    print(f"[ood-neg] wrote {len(rows)} neutral OOD transcripts -> {out}")


if __name__ == "__main__":
    main()
