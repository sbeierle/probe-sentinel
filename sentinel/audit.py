"""Closed-Loop-Audit: Monitoring → Onset → Rewind → Branches mit Kontrollen.

Branches (alle starten am SELBEN Rewind-Punkt r, mit denselben Zufallsströmen = Common Random Numbers):
    A  – beobachteter Verlauf: Phase 1 unverändert zu Ende generiert (kein Rewind)
    B  – Rewind + Ablation von v (Projektion auf Honest-Mittel geklemmt)
    C  – Rewind OHNE Eingriff  → misst Rewind-Artefakte / Nichtdeterminismus; bei greedy muss C == A gelten
    D  – Rewind + gleich starker Eingriff entlang Zufallsrichtung ⟂ v (pro Sample eine neue) → Spezifität
  (optional weitere, z. B. "E": ("add", -4.0) für Dosis-Wirkung)

Rewind-Semantik: r = erster Token, der NEU gezogen wird. Der Cache wird auf Länge r-1 gesetzt und Token r-1 erneut
eingespeist – so wirkt der Eingriff bereits auf die Verteilung von Token r.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Callable

import torch
import torch.nn.functional as F

from .cache import CacheRewinder
from .detect import Cusum
from .hooks import ResidualProbe


@dataclass
class AuditConfig:
    max_phase1: int = 1024
    max_new_tokens: int = 1024
    temperature: float = 0.0
    n_samples: int = 1  # pro Branch; bei temperature>0 sinnvoll: 8–32
    seed: int = 0
    branch_modes: dict = field(default_factory=lambda: {"B": ("ablate", 0.0), "C": ("off", 0.0), "D": ("random", 0.0)})
    rewind_margin: int = 0  # zusätzliche Tokens vor dem Onset
    snap_to_boundary: bool = True  # auf Zeilen-/Satzanfang zurücksetzen
    boundary_lookback: int = 48
    boundary_markers: tuple = ("\n", ". ", ": ")
    run_controls_without_alarm: bool = False  # Rewind an zufälliger Position auch ohne Alarm (Spezifitätsmessung)
    log_path: str | None = "audit_differential.jsonl"


def _step(model, ids: list[int], cache):
    dev = next(model.parameters()).device
    out = model(input_ids=torch.tensor([ids], device=dev), past_key_values=cache, use_cache=True)
    return out.logits[:, -1].float(), out.past_key_values


def _pick(logits, temperature, gen):
    if temperature <= 0:
        return int(logits.argmax(-1))
    probs = torch.softmax(logits / temperature, -1)
    return int(torch.multinomial(probs, 1, generator=gen))


def _gen(model, cache, first_input: int, n_max: int, temperature: float, gen, eos: set[int]):
    toks, first_logits, x = [], None, first_input
    for _ in range(n_max):
        logits, cache = _step(model, [x], cache)
        if first_logits is None:
            first_logits = logits
        x = _pick(logits, temperature, gen)
        toks.append(x)
        if x in eos:
            break
    return toks, first_logits


def _eos_ids(tok) -> set[int]:
    e = getattr(tok, "eos_token_id", None)
    if e is None:
        return set()
    return set(e) if isinstance(e, (list, tuple)) else {int(e)}


def _kl(p_logits, q_logits) -> float:
    return float(F.kl_div(F.log_softmax(q_logits, -1), F.log_softmax(p_logits, -1), log_target=True, reduction="sum"))


@torch.no_grad()
def run_audit(
    model,
    tok,
    prompt: str | list[int],
    probe: ResidualProbe,
    detector: Cusum,
    cfg: AuditConfig = AuditConfig(),
    grader: Callable[[str, str], dict] | None = None,
    rewinder: CacheRewinder | None = None,
    meta: dict | None = None,
) -> dict:
    t0 = time.time()
    eos = _eos_ids(tok)
    ids = list(prompt) if isinstance(prompt, list) else tok.encode(prompt)
    P = len(ids)
    prompt_text = prompt if isinstance(prompt, str) else tok.decode(ids)
    rewinder = rewinder or CacheRewinder()
    detector.reset()
    dev = next(model.parameters()).device
    gen = torch.Generator(device=dev).manual_seed(cfg.seed)

    # ---------------- Phase 1: Monitoring ------------------------------------------------------------------
    probe.set_mode("off")
    probe.new_trace()
    cache = None
    if P > 1:
        _, cache = _step(model, ids[:-1], cache)
        rewinder.record(cache, P - 1)
    probe.new_trace()  # Prompt-Prefill nicht in die Detektion
    z_pos, z_val = [], []
    rng_at = {}  # Position des zu ziehenden Tokens -> RNG-Zustand davor (für C == A auch bei Sampling)
    alarm = False
    while len(ids) - P < cfg.max_phase1:
        pos = len(ids) - 1
        logits, cache = _step(model, [ids[-1]], cache)
        rewinder.record(cache, len(ids))
        z = float(probe.trace[-1].flatten()[0])  # 1 Sync/Token; der Argmax synchronisiert ohnehin
        z_pos.append(pos)
        z_val.append(z)
        if cfg.temperature > 0:
            rng_at[pos + 1] = gen.get_state()
        nxt = _pick(logits, cfg.temperature, gen)
        ids.append(nxt)
        alarm = detector.update(z)
        if alarm or nxt in eos:
            break

    rec = {
        "meta": meta or {},
        "prompt": prompt_text,
        "prompt_len": P,
        "probe": {"mu": probe.mu, "sd": probe.sd, "target": probe.target},
        "detector": {"drift": detector.drift, "h": detector.h},
        "alarm": alarm,
        "config": {k: v for k, v in asdict(cfg).items() if k != "branch_modes"} | {
            "branch_modes": {k: list(v) for k, v in cfg.branch_modes.items()}},
    }

    # Rewind-Punkt bestimmen
    r = None
    if alarm:
        onset_pos = z_pos[detector.onset_at]
        rec["alarm_pos"] = z_pos[detector.alarm_at]
        rec["onset_pos"] = onset_pos
        r = onset_pos + 1 - cfg.rewind_margin
    elif cfg.run_controls_without_alarm and len(ids) > P + 2:
        r = int(torch.randint(P, len(ids) - 1, (1,), generator=torch.Generator().manual_seed(cfg.seed)))
        rec["random_rewind"] = True
    fork_cache = None
    if r is not None:
        r = max(r, P)
        if cfg.snap_to_boundary:
            for j in range(r - 1, max(P - 1, r - 1 - cfg.boundary_lookback), -1):
                if any(m in tok.decode([ids[j]]) for m in cfg.boundary_markers):
                    r = j + 1
                    break
        fork_cache, got = rewinder.fork(cache, r - 1)
        r = got + 1
        rec["rewind_pos"] = r
        rec["rewind_steps_before_alarm"] = rec.get("alarm_pos", r) - r
        rec["prefix_kept"] = tok.decode(ids[P:r])

    # ---------------- Branch A: beobachteter Verlauf zu Ende ----------------------------------------------------
    # (Fork oben hält eigene Kopien der rekurrenten Zustände; A darf den Original-Cache weiterschreiben.)
    probe.new_trace()
    if ids[-1] not in eos:
        cont, _ = _gen(model, cache, ids[-1], cfg.max_new_tokens, cfg.temperature, gen, eos)
    else:
        cont = []
    a_tokens = ids[P:] + cont
    a_trace = z_val + probe.trace_values()
    rec["A"] = {"text": tok.decode(a_tokens), "n_tokens": len(a_tokens), "trace": a_trace,
                "grade": grader(prompt_text, tok.decode(a_tokens)) if grader else None}

    # ---------------- Branches B/C/D ab r ----------------------------------------------------------------------
    if fork_cache is not None:
        rec["branches"] = {}
        base_first = None
        firsts = {}
        # gleiches absolutes Längenbudget wie A → C ist direkt mit A vergleichbar
        budget = (len(ids) - r) + cfg.max_new_tokens
        for name, (mode, coef) in cfg.branch_modes.items():
            samples = []
            for k in range(cfg.n_samples):
                # Common Random Numbers: Sample k nutzt in allen Branches denselben Zufallsstrom.
                # Sample 0 übernimmt exakt den RNG-Zustand von A an Position r → C muss A auch beim Sampling treffen.
                g = torch.Generator(device=dev).manual_seed(cfg.seed + 1000 + k)
                if k == 0 and r in rng_at:
                    g.set_state(rng_at[r])
                if mode == "random":  # pro Sample neue Kontrollrichtung → Verteilung statt Einzelvektor
                    probe.set_random_direction(cfg.seed * 7919 + k)
                c, _ = rewinder.fork(fork_cache, r - 1)
                probe.set_mode(mode, coef)
                probe.new_trace()
                toks, fl = _gen(model, c, ids[r - 1], budget, cfg.temperature, g, eos)
                probe.set_mode("off")
                if k == 0:
                    firsts[name] = fl
                full = ids[P:r] + toks
                text = tok.decode(full)
                dn = probe.delta_norms
                samples.append({
                    "seed": "A_state" if (k == 0 and r in rng_at) else cfg.seed + 1000 + k,
                    "random_dir_seed": probe.random_seed if mode == "random" else None,
                    "text": text,
                    "new_text": tok.decode(toks),
                    "n_tokens": len(full),
                    "trace": probe.trace_values(),
                    "mean_delta_norm": float(torch.stack(dn).mean()) if dn else 0.0,
                    "grade": grader(prompt_text, text) if grader else None,
                    "_toks": toks,
                })
            rec["branches"][name] = samples
            if mode == "off":
                base_first = firsts[name]
        # lokale Effektstärke am Divergenzpunkt: KL(Branch || unverändert) für Token r
        if base_first is not None:
            rec["first_token_kl_vs_C"] = {n: _kl(fl, base_first) for n, fl in firsts.items()}
        # Determinismus-Check: reproduziert C (Sample 0) den beobachteten Verlauf A?
        # Greedy: muss exakt gelten. Sampling: gilt exakt, weil Sample 0 den RNG-Zustand von A übernimmt.
        # Abweichungen = Kernel-Nichtdeterminismus oder Rewind-Fehler → als Metrik berichten, nicht verstecken.
        if "C" in rec["branches"]:
            c_toks = rec["branches"]["C"][0]["_toks"]
            a_tail = a_tokens[r - P:]
            m = min(len(c_toks), len(a_tail))
            first_div = next((i for i in range(m) if c_toks[i] != a_tail[i]), None)
            if first_div is None and len(c_toks) != len(a_tail):
                first_div = m
            agree = sum(c_toks[i] == a_tail[i] for i in range(m)) / max(1, max(len(c_toks), len(a_tail)))
            rec["C_reproduces_A"] = {"C_equals_A_exact": c_toks == a_tail, "C_token_agreement": round(agree, 4),
                                     "C_first_divergence_token": first_div,
                                     "len_C": len(c_toks), "len_A_tail": len(a_tail)}
        for s in rec["branches"].values():
            for x in s:
                x.pop("_toks", None)

    rec["seconds"] = round(time.time() - t0, 3)
    rec["hybrid"] = bool(rewinder.hybrid)
    if rec["hybrid"]:
        rec["hybrid_snapshot_bytes_per_step"] = rewinder.snapshot_bytes
    if cfg.log_path:
        with open(cfg.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec
