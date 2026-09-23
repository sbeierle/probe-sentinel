"""Residual-Stream-Probe am Ausgang eines Decoder-Layers.

Änderungen gegenüber dem Entwurf:
  * Tensor- UND Tupel-Rückgaben der Decoder-Layer werden unterstützt (transformers >= 5 gibt einen Tensor zurück;
    `output[0]` wäre dort die erste Batch-Zeile, der alte Hook bricht still).
  * Batch-fähig: Projektion per Matmul statt `torch.dot`.
  * Scores werden z-normalisiert gegen eine Honest-Baseline (mu, sd) statt absoluter Schwelle.
  * Trace pro Branch getrennt (`new_trace()`), keine Vermischung zwischen Phase 1 und Branches.
  * Steering-Modi:
        off     – nur messen
        ablate  – Projektion auf den Honest-Mittelwert klemmen: h <- h - (s - target) * v   (kein Überschießen)
        random  – identische Eingriffsstärke |s - target| pro Token, aber entlang eines zufälligen, zu v orthogonalen
                  Einheitsvektors u (Kontrolle D), optional im Unterraum der tatsächlichen Aktivierungen
        add     – h <- h + c * sd * v   (Dosis-Wirkungs-Kurven, c negativ = Richtung "honest")
  * Gemessen wird VOR dem Eingriff (was das Modell "wollte"); die tatsächlich angewandte Eingriffsnorm wird mitgeloggt.
"""
from __future__ import annotations

import torch
from torch import nn

MODES = ("off", "ablate", "random", "add")


def get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Findet die Liste der Decoder-Layer (Llama/Qwen/Mixtral/Qwen3.5-VL-Wrapper ...)."""
    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and name.endswith("layers") and "vision" not in name and "visual" not in name:
            if best is None or len(mod) > len(best):
                best = mod
    if best is None:
        raise ValueError("Keine Decoder-Layer-Liste gefunden")
    return best


class ResidualProbe:
    def __init__(
        self,
        layer: nn.Module,
        v: torch.Tensor,
        mu: float,
        sd: float,
        target: float | None = None,
        seed: int = 0,
        random_basis: torch.Tensor | None = None,
    ):
        v = v.detach().float().flatten()
        self.v_cpu = v / v.norm()
        self.basis = None if random_basis is None else random_basis.detach().float().cpu()  # (D, k)
        self.set_random_direction(seed)
        self.mu, self.sd = float(mu), float(sd)
        self.target = float(mu if target is None else target)
        self.mode = "off"
        self.add_coef = 0.0
        self.trace: list[torch.Tensor] = []  # z-Scores (letzte Position) pro Forward, vor Eingriff
        self.delta_norms: list[torch.Tensor] = []  # Norm des Eingriffs pro Forward
        self.handle = layer.register_forward_hook(self._hook)

    # -- Verwaltung ---------------------------------------------------------------------------------
    def set_mode(self, mode: str, add_coef: float = 0.0) -> None:
        if mode not in MODES:
            raise ValueError(f"mode muss in {MODES} sein")
        self.mode, self.add_coef = mode, add_coef

    def set_random_direction(self, seed: int) -> None:
        """Kontrollrichtung u für Branch D: zufällig, ⟂ v, Einheitsnorm.
        Mit `random_basis` (z. B. skalierte Top-PCs ehrlicher Aktivierungen, siehe ProbeArtifact.pcs) wird u im
        tatsächlich genutzten Aktivierungs-Unterraum gezogen. Isotrope Zufallsrichtungen liegen überwiegend in
        Dimensionen mit kaum Varianz und sind deshalb eine zu schwache Kontrolle."""
        # eigener Seed-Raum: sonst kann u zufällig mit einem gleich geseedeten v zusammenfallen (u ⟂ v → Nullvektor)
        g = torch.Generator().manual_seed(0x5EED0000 + seed)
        for _ in range(8):
            if self.basis is None:
                u = torch.randn(self.v_cpu.shape, generator=g)
            else:
                u = self.basis @ torch.randn(self.basis.shape[1], generator=g)
            raw = u.norm()
            u = u - (u @ self.v_cpu) * self.v_cpu
            if raw > 0 and u.norm() > 1e-4 * raw:  # relativ prüfen: die Basis-Skala hängt vom Modell ab
                break
        else:
            raise ValueError("Keine Kontrollrichtung ⟂ v in der Basis gefunden (Basis ⊂ span(v)?)")
        self.u_cpu = u / u.norm()
        self.random_seed = seed
        self._dev_cache = {}

    def new_trace(self) -> None:
        self.trace, self.delta_norms = [], []

    def trace_values(self) -> list[float]:
        """Eine Synchronisation am Ende statt `.item()` pro Token."""
        if not self.trace:
            return []
        return torch.stack([t.flatten()[0] for t in self.trace]).cpu().tolist()

    def remove(self) -> None:
        self.handle.remove()

    def _vecs(self, h: torch.Tensor):
        key = h.device
        if key not in self._dev_cache:
            self._dev_cache[key] = (self.v_cpu.to(h.device), self.u_cpu.to(h.device))
        return self._dev_cache[key]

    # -- Hook ---------------------------------------------------------------------------------------
    def _hook(self, module, args, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output  # (B, T, D)
        v, u = self._vecs(h)
        hf = h.float()
        s = hf @ v  # (B, T) Rohprojektion
        self.trace.append(((s[:, -1] - self.mu) / self.sd).detach())

        if self.mode == "off":
            self.delta_norms.append(torch.zeros((), device=h.device))
            return None
        if self.mode == "ablate":
            delta = -(s - self.target).unsqueeze(-1) * v
        elif self.mode == "random":
            delta = -(s - self.target).unsqueeze(-1) * u
        else:  # add
            delta = (self.add_coef * self.sd) * v.expand_as(hf)
        self.delta_norms.append(delta[:, -1].norm(dim=-1).mean().detach())
        h_new = (hf + delta).to(h.dtype)
        return (h_new,) + tuple(output[1:]) if is_tuple else h_new
