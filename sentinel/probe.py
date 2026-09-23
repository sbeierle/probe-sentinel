"""Probe-Extraktion mit Validierung.

Kernpunkte gegenüber dem Entwurf:
  * ON-POLICY-Daten: Beispiele sind (prompt, completion, label, group), wobei completion vom Zielmodell selbst
    stammt und label vom Grader kommt (hat gehackt / hat ehrlich gelöst) – nicht handgeschriebene Kontrastpaare.
  * Gruppierte Kreuzvalidierung nach `group` (= Task-ID): sonst lernt der Probe die Aufgabe, nicht das Verhalten.
  * Layer-Sweep statt fix L26, mit Kontrolle "geshuffelte Labels" (muss ~0.5 AUROC ergeben).
  * Kalibrierung: mu/sd der Projektion auf Token-Ebene über ehrliche Completions, erste Positionen
    (Attention Sinks / Massive Activations) werden ausgeschlossen.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import torch

from .hooks import get_decoder_layers


@dataclass
class ProbeArtifact:
    layer: int
    v: torch.Tensor
    mu: float  # Mittel der Token-Projektion auf ehrlichen Completions
    sd: float
    cv_auroc: float
    meta: dict
    pcs: torch.Tensor | None = None  # (D, k) skalierte Top-PCs ehrlicher Aktivierungen → Basis für Kontrolle D

    def save(self, path: str) -> None:
        torch.save({**asdict(self), "v": self.v.cpu(), "pcs": None if self.pcs is None else self.pcs.cpu()}, path)

    @classmethod
    def load(cls, path: str) -> "ProbeArtifact":
        return cls(**torch.load(path, weights_only=False))


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rangbasierte AUROC (Mann-Whitney), ohne sklearn."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    order = scores.argsort()
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    # Ties mitteln
    for val in np.unique(scores):
        m = scores == val
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    n1, n0 = labels.sum(), (1 - labels).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


@torch.no_grad()
def collect_activations(model, tok, examples: list[dict], layers: list[int], skip_first: int = 4,
                        max_tokens_per_example: int = 2048, predictive: bool = False,
                        token_layers: list[int] | None = None):
    """Liefert pro Layer: pooled [N, D] (Mittel über Completion-Tokens) und per-Token-Listen.
    examples: {"prompt": str, "completion": str, "label": 0|1, "group": hashable}
    predictive=True: Token-Liste beginnt beim letzten Prompt-Token, d. h. Eintrag i ist der Zustand, aus dem
    Completion-Token i vorhergesagt wird (für Lead-Time-Messungen: "wusste es das Modell, bevor es schrieb?").
    token_layers: nur für diese Layer Token-Aktivierungen behalten (RAM! 7B × 10 Layer × 300 Tokens × 256 Beispiele
    wären ~11 GB). Default: alle `layers`."""
    token_layers = layers if token_layers is None else token_layers
    dec = get_decoder_layers(model)
    buf: dict[int, torch.Tensor] = {}
    hooks = []
    for L in layers:
        def mk(L):
            def f(m, a, o):
                buf[L] = (o[0] if isinstance(o, tuple) else o).detach().float()
            return f
        hooks.append(dec[L].register_forward_hook(mk(L)))
    dev = next(model.parameters()).device
    pooled = {L: [] for L in layers}
    tokens = {L: [] for L in layers}
    try:
        for ex in examples:
            # Bevorzugt die exakten Token-IDs aus der Generierung (on-policy, keine Re-Tokenisierungsartefakte)
            p_ids = list(ex["prompt_ids"]) if "prompt_ids" in ex else tok.encode(ex["prompt"])
            if "completion_ids" in ex:
                c_ids = list(ex["completion_ids"])
            else:
                c_ids = tok.encode(ex["completion"], add_special_tokens=False) if _hf(tok) else tok.encode(ex["completion"])
            ids = (p_ids + c_ids)[:max_tokens_per_example]
            model(torch.tensor([ids], device=dev), use_cache=False)
            if predictive:
                start, stop = len(p_ids) - 1, len(p_ids) - 1 + len(c_ids)
            else:
                start, stop = max(len(p_ids), skip_first), None
                if ex.get("cut") is not None:  # nur Zustände vor dem ersten Hack-Marker (Prefix-Probe)
                    stop = max(start + 1, len(p_ids) + int(ex["cut"]))
            for L in layers:
                h = buf[L][0, start:stop].cpu()
                if h.shape[0] == 0:  # leere Completion (sofort EOS): letzter verfügbarer Zustand statt NaN-Mittel
                    h = buf[L][0, -1:].cpu()
                pooled[L].append(h.mean(0))
                if L in token_layers:
                    tokens[L].append(h)
    finally:
        for h in hooks:
            h.remove()
    return {L: torch.stack(pooled[L]) for L in layers}, tokens


def _hf(tok) -> bool:
    return hasattr(tok, "pad_token_id")


def _group_folds(groups, k):
    uniq = np.unique(groups)
    rng = np.random.default_rng(0)
    rng.shuffle(uniq)
    return [np.isin(groups, part) for part in np.array_split(uniq, k)]


def _dom(X, y):
    v = X[y == 1].mean(0) - X[y == 0].mean(0)
    return v / (v.norm() + 1e-8)


def cv_auroc(X: torch.Tensor, y: np.ndarray, groups: np.ndarray, k: int = 5) -> float:
    k = min(k, len(np.unique(groups)))
    preds = np.zeros(len(y))
    for test in _group_folds(groups, k):
        tr = ~test
        v = _dom(X[torch.from_numpy(tr)], y[tr])
        preds[test] = (X[torch.from_numpy(test)] @ v).numpy()
    return auroc(preds, y)


def layer_sweep(pooled: dict, labels, groups, k: int = 5) -> list[dict]:
    y, g = np.asarray(labels), np.asarray(groups)
    rng = np.random.default_rng(1)
    rows = []
    for L, X in pooled.items():
        rows.append({"layer": L, "cv_auroc": cv_auroc(X, y, g, k),
                     "shuffled_control": cv_auroc(X, rng.permutation(y), g, k)})
    return sorted(rows, key=lambda r: -r["cv_auroc"])


def fit_probe(pooled: dict, tokens: dict, labels, groups, layer: int, skip: int = 0, meta: dict | None = None,
              n_pcs: int = 64) -> ProbeArtifact:
    y, g = np.asarray(labels), np.asarray(groups)
    X = pooled[layer]
    v = _dom(X, y)
    if not torch.isfinite(v).all():
        raise ValueError("Probe-Richtung nicht endlich – leere/degenerierte Completions in den Trainingsdaten?")
    honest_tok = torch.cat([t[skip:] for t, lab in zip(tokens[layer], y) if lab == 0])
    proj = honest_tok @ v
    c = honest_tok - honest_tok.mean(0)
    k = min(n_pcs, c.shape[0] - 1, c.shape[1])
    pcs = None  # zu wenige ehrliche Tokens → Kontrolle D fällt auf isotrope Richtungen zurück
    if k >= 2:
        _, S, Vh = torch.linalg.svd(c, full_matrices=False)
        pcs = (Vh[:k].T * (S[:k] / max(1, c.shape[0] - 1) ** 0.5))  # Spalten = PC-Richtung × Std-Abweichung
    return ProbeArtifact(layer=layer, v=v, mu=float(proj.mean()), sd=float(proj.std() + 1e-6),
                         cv_auroc=cv_auroc(X, y, g), meta=meta or {}, pcs=pcs)


def cross_type_auroc(X: torch.Tensor, labels, types, groups, min_n: int = 5) -> dict:
    """Train auf EINEM Hack-Typ (+ ehrliche Lösungen aus der Hälfte der Gruppen), Test auf den ANDEREN Hack-Typen
    (+ ehrliche aus der anderen Hälfte). Bricht die AUROC ein, ist der Probe ein Muster-Detektor."""
    y, t, g = np.asarray(labels), np.asarray(types), np.asarray(groups)
    fold_a = _group_folds(g, 2)[0]
    out = {}
    for typ in sorted(set(t[y == 1])):
        tr = ((y == 1) & (t == typ)) | ((y == 0) & fold_a)
        te = ((y == 1) & (t != typ)) | ((y == 0) & ~fold_a)
        if ((y == 1) & (t == typ)).sum() < min_n or ((y == 1) & te).sum() < min_n:
            continue
        v = _dom(X[torch.from_numpy(tr)], y[tr])
        out[typ] = {"auroc_other_types": auroc((X[torch.from_numpy(te)] @ v).numpy(), y[te]),
                    "n_train_hacks": int(((y == 1) & tr).sum()), "n_test_hacks": int(((y == 1) & te).sum())}
    return out
