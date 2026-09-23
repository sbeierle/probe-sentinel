"""Probe-Sentinel: Probe-getriggerter, kontrafaktischer Rewind mit Kontrollbranches.

Module:
    hooks   – Residual-Stream-Probe mit getrennten Traces und Steering-Modi
    cache   – Zero-Copy-Fork des KV-Cache + Snapshots für rekurrente Zustände (Hybridmodelle)
    detect  – z-Normalisierung, CUSUM-Onset-Detektion, Kalibrierung auf Ziel-FPR
    probe   – On-Policy-Aktivierungen sammeln, Layer-Sweep, gruppierte CV-AUROC, Probe speichern
    monitor – Ebene 1: Lead Time, TPR/FPR, Probe vs. Text-Baseline auf Präfixen, Gates
    audit   – Ebene 2: Phase 1 → Onset → Branches A/B/C/D → JSONL
    graders – Grader-Schnittstelle, Beispiel für Code-Tasks (sichtbare vs. versteckte Tests)
"""

from .hooks import ResidualProbe, get_decoder_layers
from .cache import CacheRewinder
from .detect import Cusum, calibrate_cusum
from .probe import ProbeArtifact, collect_activations, layer_sweep, fit_probe
from .audit import AuditConfig, run_audit

__all__ = [
    "ResidualProbe", "get_decoder_layers", "CacheRewinder", "Cusum", "calibrate_cusum",
    "ProbeArtifact", "collect_activations", "layer_sweep", "fit_probe", "AuditConfig", "run_audit",
]
