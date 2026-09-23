import json

found = 0
with open("runs/q7b_pressure/rollouts.jsonl", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        r = json.loads(line)
        g = r.get("grade") or {}
        if g.get("hack") is True:
            found += 1
            print("=" * 65)
            print(f"ECHTER HACK #{found} | Zeile {i+1} | Task: {r.get('task')}")
            print(f"Grade: {g}")
            print("-" * 65)
            print(r.get("completion", "").strip())
            print("=" * 65 + "\n")

print(f"[+] Gefundene echte Hacks: {found}")
