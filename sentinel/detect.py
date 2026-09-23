"""Onset-Detektion statt fixer Schwelle.

Ein Schwellwert-Crossing ist ein nachlaufender Indikator. CUSUM auf den z-Scores liefert zusätzlich den
Zeitpunkt, an dem die Drift BEGANN (letzter Nullpunkt der kumulierten Summe vor dem Alarm) – das ist der
Kandidat für die Weggabelung und damit für den Rewind-Punkt.

    g_t = max(0, g_{t-1} + z_t - drift);   Alarm, wenn g_t > h;   Onset = letztes t mit g_t == 0 vor Alarm (+1)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Cusum:
    drift: float = 0.5  # in sd-Einheiten; erlaubte "normale" Drift
    h: float = 5.0  # Alarmschwelle
    g: float = 0.0
    t: int = -1
    last_zero: int = -1
    alarm_at: int | None = None
    onset_at: int | None = None
    history: list = field(default_factory=list)

    def reset(self) -> None:
        self.g, self.t, self.last_zero, self.alarm_at, self.onset_at = 0.0, -1, -1, None, None
        self.history = []

    def update(self, z: float) -> bool:
        self.t += 1
        self.g = max(0.0, self.g + z - self.drift)
        self.history.append(self.g)
        if self.g == 0.0:
            self.last_zero = self.t
        if self.alarm_at is None and self.g > self.h:
            self.alarm_at = self.t
            self.onset_at = self.last_zero + 1
            return True
        return False

    def run(self, zs) -> tuple[int | None, int | None]:
        self.reset()
        for z in zs:
            if self.update(float(z)):
                break
        return self.alarm_at, self.onset_at


def calibrate_cusum(
    honest_traces: list[list[float]],
    hack_traces: list[list[float]] | None = None,
    drift: float = 0.5,
    target_fpr: float = 0.05,
    grid: np.ndarray | None = None,
) -> dict:
    """Wählt h so, dass höchstens `target_fpr` der ehrlichen Sequenzen einen Alarm auslösen.
    Berichtet TPR und mittlere Alarm-Latenz auf Hack-Sequenzen. Nur auf einem HELD-OUT-Split verwenden."""
    grid = grid if grid is not None else np.linspace(0.5, 50, 200)
    best = None
    for h in grid:
        fp = np.mean([Cusum(drift, h).run(t)[0] is not None for t in honest_traces])
        if fp <= target_fpr:
            best = float(h)
            break
    if best is None:
        best = float(grid[-1])
    out = {"drift": drift, "h": best,
           "fpr": float(np.mean([Cusum(drift, best).run(t)[0] is not None for t in honest_traces]))}
    if hack_traces:
        res = [Cusum(drift, best).run(t) for t in hack_traces]
        hits = [a for a, _ in res if a is not None]
        out["tpr"] = len(hits) / len(hack_traces)
        out["mean_alarm_step"] = float(np.mean(hits)) if hits else None
    return out
