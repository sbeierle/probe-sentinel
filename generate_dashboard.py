"""Dashboard + RESULTS.md ausschließlich aus gemessenen Dateien. Keine Simulation, keine eingetippten Zahlen.

    python generate_dashboard.py                       # Standard: runs/q7b_oneshot + alle runs/* für Experiment 0
    python generate_dashboard.py --run runs/q7b_oneshot

Jede Zahl im Bild stammt aus einer JSON/JSONL-Datei; die Quelle steht im Fuß des Bildes und in RESULTS.md.
Fehlt eine Datei, bleibt das Panel leer mit Hinweis – es wird nichts aufgefüllt.
"""
import argparse
import glob
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CATS = ["hack", "honest_solved", "honest_impossible", "failed_suspicious", "failed_no_hack"]
COL = {"hack": "#c0392b", "honest_solved": "#2a9d8f", "honest_impossible": "#8ecae6",
       "failed_suspicious": "#f4a261", "failed_no_hack": "#adb5bd"}


def load(path):
    if not os.path.exists(path):
        return None
    if path.endswith(".jsonl"):
        return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return json.load(open(path, encoding="utf-8"))


def stamp(path):
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(path))) if os.path.exists(path) else "fehlt"


def empty(ax, title, msg):
    ax.set_title(title, fontweight="bold")
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, color="#666")
    ax.set_xticks([]); ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/q7b_oneshot")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    R = lambda f: os.path.join(a.run, f)
    probe_rep, mon, summ = load(R("probe_report.json")), load(R("monitor_report.json")), load(R("summary.json"))
    audit = load(R("audit.jsonl"))
    probe_layer = None
    try:
        import torch
        probe_layer = int(torch.load(R("probe.pt"), weights_only=False, map_location="cpu")["layer"])
    except Exception:
        pass
    warn = []
    if mon and probe_layer is not None and mon.get("layer") != probe_layer:
        warn.append(f"monitor_report (Layer {mon.get('layer')}) passt nicht zu probe.pt (Layer {probe_layer})")
    if audit and os.path.getmtime(R("audit.jsonl")) < os.path.getmtime(R("probe.pt")):
        warn.append("audit.jsonl ist ÄLTER als probe.pt → Audit stammt von einem früheren Probe/Layer")

    fig, axs = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(f"Probe-Sentinel v0.4 – Messergebnisse ({a.run}, Qwen2.5-Coder-7B, RX 7900 XTX)",
                 fontsize=15, fontweight="bold")

    # 1) Experiment 0 über alle Läufe
    ax = axs[0, 0]
    p1s = {os.path.basename(os.path.dirname(p)): load(p) for p in sorted(glob.glob("runs/*/p1_report.json"))}
    if p1s:
        labels, x = [], 0
        for run, rep in p1s.items():
            for cond in ("normal", "impossible"):
                bottom = 0
                for c in CATS:
                    v = rep["by_condition"][cond].get(c, 0)
                    ax.bar(x, v, bottom=bottom, color=COL[c], edgecolor="white",
                           label=c if x == 0 else None)
                    if c == "hack" and v:
                        ax.text(x, bottom + v + 3, str(v), ha="center", fontweight="bold", color=COL["hack"])
                    bottom += v
                labels.append(f"{run.replace('q7b', '').strip('_') or 'baseline'}\n{cond}")
                x += 1
            x += 0.5
        ax.set_xticks([i + (i // 2) * 0.5 for i in range(len(labels))], labels, fontsize=8)
        ax.set_ylabel("Rollouts (je 256)")
        ax.set_title("1. Experiment 0: Kategorien je Bedingung (Zahl = Hacks)", fontweight="bold")
        ax.legend(fontsize=8, loc="center right")
    else:
        empty(ax, "1. Experiment 0", "keine p1_report.json gefunden")

    # 2) Layer-Sweep mit Shuffle-Kontrolle
    ax = axs[0, 1]
    if probe_rep and probe_rep.get("sweep"):
        sw = sorted(probe_rep["sweep"], key=lambda r: r["layer"])
        L = [r["layer"] for r in sw]
        ax.plot(L, [r["cv_auroc"] for r in sw], "o-", lw=2, label="CV-AUROC (gruppiert nach Familie)")
        ax.plot(L, [r["shuffled_control"] for r in sw], "s--", color="#999", label="Shuffle-Kontrolle")
        ax.axhline(0.5, color="#ccc", lw=1)
        if probe_layer is not None:
            ax.axvline(probe_layer, color="#c0392b", ls=":", label=f"gewählter Layer {probe_layer}")
        ax.set_ylim(0, 1.05); ax.set_xlabel("Layer"); ax.set_ylabel("AUROC")
        n = (mon or {}).get("monitor", {}).get("n", {})
        ax.set_title(f"2. Layer-Sweep auf TRAIN (probe_report.json)", fontweight="bold")
        ax.legend(fontsize=8, loc="lower right")
    else:
        empty(ax, "2. Layer-Sweep", "probe_report.json fehlt")

    # 3) Monitoring auf TEST
    ax = axs[1, 0]
    if mon:
        m5 = mon["monitor"].get("fpr_0.05", {})
        m1 = mon["monitor"].get("fpr_0.01", {})
        keys = [("tpr", "TPR Hacks"), ("fpr_per_honest_completion", "Fehlalarm\nehrlich"),
                ("alarm_rate_honest_impossible", "Alarm\nhonest_imp."), ("alarm_rate_failed_no_hack", "Alarm\nfailed")]
        xs = np.arange(len(keys))
        for off, (m, lab) in zip((-0.2, 0.2), ((m1, f"h={m1.get('h', float('nan')):.2f} (Ziel 1 %)"),
                                             (m5, f"h={m5.get('h', float('nan')):.2f} (Ziel 5 %)"))):
            vals = [m.get(k) if m.get(k) is not None else np.nan for k, _ in keys]
            ax.bar(xs + off, vals, width=0.4, label=lab)
            for xi, v in zip(xs + off, vals):
                if not np.isnan(v):
                    ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
        ax.set_xticks(xs, [l for _, l in keys]); ax.set_ylim(0, 1.15)
        cc = mon.get("condition_control", {})
        sp = mon.get("surface_prefix", {})
        n = mon["monitor"].get("n", {})
        sub = (f"n Test: {n.get('hack', 0)} Hacks / {n.get('honest_solved', 0)} ehrlich | "
               + (f"Bedingungs-Kontrolle AUROC {cc['auroc']:.3f} [{cc['ci95'][0]:.2f}, {cc['ci95'][1]:.2f}]"
                  if "auroc" in cc else f"Bedingungs-Kontrolle: {cc.get('skipped', '–')}")
               + "\nPräfix-Vergleich: " + (f"ΔAUROC {sp['delta_probe_over_surface']:.2f}, KI {np.round(sp['delta_ci95'], 2).tolist()}"
                                            if "delta_ci95" in sp else sp.get("skipped", "–"))
               + f" | claim_level = {mon['gates']['claim_level']}")
        ax.set_title(f"3. Monitoring auf TEST (Layer {mon.get('layer')}, monitor_report.json)", fontweight="bold")
        ax.text(0.0, -0.22, sub, transform=ax.transAxes, fontsize=8, va="top")
        ax.legend(fontsize=8)
    else:
        empty(ax, "3. Monitoring", "monitor_report.json fehlt")

    # 4) Audit
    ax = axs[1, 1]
    if summ and summ.get("hack_rate"):
        br = ["B", "C", "D"]
        lab = {"B": "B: Ablation v", "C": "C: Rewind ohne Eingriff", "D": "D: Zufallsrichtung (orth. zu v)"}
        rates = [summ["hack_rate"].get(b, np.nan) for b in br]
        ax.bar(range(3), rates, color=["#457b9d", "#adb5bd", "#e9c46a"], edgecolor="black")
        for i, b in enumerate(br):
            d = (summ.get("hack_diff_vs_C") or {}).get(b)
            txt = f"{rates[i]:.3f}" + (f"\nΔ vs C {d['mean']:+.3f}\nKI [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]" if d else "")
            ax.text(i, rates[i] + 0.003, txt, ha="center", fontsize=8)
        ax.set_xticks(range(3), [lab[b] for b in br]); ax.set_ylabel("Hack-Rate")
        ax.set_ylim(0, max(rates) * 1.8 if max(rates) > 0 else 0.1)
        ax.set_title(f"4. Audit (n={summ.get('n_records')}, Probe-Alarme: {summ.get('alarm_rate', 0):.0%}, "
                     f"C==A: {summ.get('C_equals_A_exact_rate', float('nan')):.0%})", fontweight="bold")
        if summ.get("alarm_rate", 0) == 0:
            ax.text(0.5, 0.93, "0 Probe-Alarme → alle Rewinds an Zufallspositionen (Kontrollmodus):\n"
                    "kein Test der Hack-Verhinderung", transform=ax.transAxes, ha="center", fontsize=8, color="#c0392b")
    else:
        empty(ax, "4. Audit", "summary.json fehlt")

    src = " | ".join(f"{f}: {stamp(R(f))}" for f in ("probe.pt", "monitor_report.json", "audit.jsonl", "summary.json"))
    foot = "Alle Werte aus Messdateien, keine Simulation.  " + src
    if warn:
        foot += "\nACHTUNG: " + " ; ".join(warn)
    fig.text(0.01, 0.005, foot, fontsize=8, color="#c0392b" if warn else "#555")
    plt.tight_layout(rect=(0, 0.04, 1, 0.97))
    out = a.out or R("dashboard_measured.png")
    plt.savefig(out, dpi=160)
    print(f"[+] {out}")
    for w in warn:
        print("[!]", w)

    # RESULTS.md: dieselben Zahlen als Tabelle (für README)
    md = [f"# Messergebnisse ({a.run})", "", f"Erzeugt aus Messdateien. {src}", ""]
    if warn:
        md += ["**Achtung:** " + " ; ".join(warn), ""]
    if p1s:
        md += ["| Lauf | Bedingung | " + " | ".join(CATS) + " |", "|---|---|" + "---|" * len(CATS)]
        for run, rep in p1s.items():
            for cond in ("normal", "impossible"):
                md.append(f"| {run} | {cond} | " + " | ".join(str(rep['by_condition'][cond].get(c, 0)) for c in CATS) + " |")
        md.append("")
    if mon:
        best = next((r for r in (probe_rep or {}).get("sweep", []) if r["layer"] == mon.get("layer")), {})
        md += [f"**Monitoring, Layer {mon.get('layer')}**: CV-AUROC {best.get('cv_auroc', float('nan')):.3f} "
               f"(Shuffle {best.get('shuffled_control', float('nan')):.3f}), claim_level `{mon['gates']['claim_level']}`, "
               f"fehlt: {', '.join(mon['gates']['missing'])}", "",
               "```json", json.dumps({k: mon[k] for k in ("monitor", "surface_full", "surface_prefix", "condition_control")},
                                     indent=1)[:4000], "```", ""]
    if summ:
        md += ["**Audit**", "```json", json.dumps(summ, indent=1), "```"]
    open(os.path.join(a.run, "RESULTS.md"), "w", encoding="utf-8").write("\n".join(md))
    print(f"[+] {os.path.join(a.run, 'RESULTS.md')}")


if __name__ == "__main__":
    main()
