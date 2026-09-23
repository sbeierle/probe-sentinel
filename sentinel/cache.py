"""Rewind/Fork des Cache.

Reine Attention (DynamicCache):
    `DynamicLayer.update` verwendet `torch.cat` → alte Tensoren werden nie in-place verändert.
    Ein Fork ist daher ZERO-COPY: Keys/Values werden per Slice-View auf die Rewind-Länge gekürzt, jeder Branch
    hängt eigene neue Tensoren an. Kein Ringpuffer mit Deepcopies nötig.

Hybrid (Qwen3.5 / Qwen3-Next: Gated DeltaNet + Attention):
    Rekurrente Zustände (conv_states, recurrent_states) sind NICHT append-only und werden teils in-place
    aktualisiert (fused conv-Kernel im Single-Token-Decode). Sie lassen sich nicht zurückrechnen.
    → Wir halten pro Schritt (oder alle `stride` Schritte) Klone NUR dieser Zustände. Die sind pro Layer konstant groß
      (unabhängig von der Sequenzlänge). Rewind = Attention-Teil slicen + rekurrenten Teil aus Snapshot klonen.

Nicht unterstützt (explizit abgefangen): StaticCache (in-place), Sliding-Window-Layer über die Fensterlänge hinaus,
Offloaded/Quantized Caches.
"""
from __future__ import annotations

import copy
from collections import OrderedDict

import torch


def _is_attn(layer) -> bool:
    return hasattr(layer, "keys") and hasattr(layer, "values")


def _is_linear(layer) -> bool:
    return hasattr(layer, "recurrent_states") or hasattr(layer, "conv_states")


def _clone_states(obj):
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    if isinstance(obj, dict):
        return {k: _clone_states(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_clone_states(v) for v in obj)
    return copy.deepcopy(obj)


def _state_bytes(obj) -> int:
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(_state_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_state_bytes(v) for v in obj)
    return 0


class CacheRewinder:
    """Hält rekurrente Snapshots (nur bei Hybridmodellen) und erzeugt Forks auf eine gegebene Cache-Länge."""

    LINEAR_ATTRS = ("conv_states", "recurrent_states")

    def __init__(self, stride: int = 1, max_snapshots: int = 256):
        self.stride = stride
        self.max_snapshots = max_snapshots
        self.snaps: OrderedDict[int, dict] = OrderedDict()
        self.hybrid: bool | None = None
        self.snapshot_bytes = 0

    @staticmethod
    def seq_len(cache) -> int:
        if hasattr(cache, "layers"):
            for layer in cache.layers:
                if _is_attn(layer) and getattr(layer, "keys", None) is not None and layer.keys.numel() > 0:
                    return layer.keys.shape[-2]
        if hasattr(cache, "key_cache"):
            for k in cache.key_cache:
                if isinstance(k, torch.Tensor) and k.numel() > 0:
                    return k.shape[-2]
        if hasattr(cache, "get_seq_length"):
            try:
                return cache.get_seq_length()
            except TypeError:
                try:
                    return cache.get_seq_length(0)
                except Exception:
                    pass
            except Exception:
                pass
        return 0

    def _check(self, cache) -> None:
        if self.hybrid is None:
            if hasattr(cache, "conv_states") or hasattr(cache, "recurrent_states"):
                cs = getattr(cache, "conv_states", None)
                rs = getattr(cache, "recurrent_states", None)
                has_cs = cs is not None and any(x is not None for x in cs)
                has_rs = rs is not None and any(x is not None for x in rs)
                self.hybrid = bool(has_cs or has_rs)
            elif hasattr(cache, "layers"):
                self.hybrid = any(_is_linear(l) for l in cache.layers)
            else:
                self.hybrid = False

        if hasattr(cache, "layers"):
            for l in cache.layers:
                name = type(l).__name__
                if "Static" in name or "Offloaded" in name or "Quantized" in name:
                    raise NotImplementedError(f"Cache-Layer {name} wird in-place geschrieben; bitte DynamicCache verwenden")
        name = type(cache).__name__
        if "Static" in name or "Offloaded" in name or "Quantized" in name:
            raise NotImplementedError(f"Cache {name} wird in-place geschrieben; bitte DynamicCache verwenden")

    def record(self, cache, length: int) -> None:
        """Nach jedem Forward aufrufen. `length` = Anzahl bereits verarbeiteter Positionen."""
        self._check(cache)
        if not self.hybrid or length % self.stride:
            return
        snap = {}
        if hasattr(cache, "layers"):
            for i, l in enumerate(cache.layers):
                if _is_linear(l):
                    snap[i] = {a: _clone_states(getattr(l, a)) for a in self.LINEAR_ATTRS if hasattr(l, a)}
        elif hasattr(cache, "conv_states") or hasattr(cache, "recurrent_states"):
            cs = getattr(cache, "conv_states", []) or []
            rs = getattr(cache, "recurrent_states", []) or []
            n_layers = max(len(cs), len(rs))
            for i in range(n_layers):
                layer_snap = {}
                if i < len(cs) and cs[i] is not None:
                    layer_snap["conv_states"] = _clone_states(cs[i])
                if i < len(rs) and rs[i] is not None:
                    layer_snap["recurrent_states"] = _clone_states(rs[i])
                if layer_snap:
                    snap[i] = layer_snap
        self.snaps[length] = snap
        if self.snapshot_bytes == 0:
            self.snapshot_bytes = _state_bytes(snap)
        while len(self.snaps) > self.max_snapshots:
            self.snaps.popitem(last=False)

    def resolve(self, length: int) -> int:
        """Größte erreichbare Cache-Länge <= length (bei Attention-only immer length selbst)."""
        if not self.hybrid:
            return length
        cands = [p for p in self.snaps if p <= length]
        if not cands:
            raise ValueError(
                f"Kein Snapshot <= {length}. Ältester: {next(iter(self.snaps), None)}. max_snapshots/stride erhöhen."
            )
        return max(cands)

    def fork(self, cache, length: int):
        """Neuer, unabhängiger Cache mit genau `length` Positionen. Original bleibt unverändert."""
        length = self.resolve(length)
        memo = {}
        if hasattr(cache, "layers"):
            for l in cache.layers:  # große KV-Tensoren NICHT kopieren (Zero-Copy)
                if _is_attn(l):
                    for t in (l.keys, l.values):
                        if isinstance(t, torch.Tensor):
                            memo[id(t)] = t
            c = copy.deepcopy(cache, memo)
            for i, l in enumerate(c.layers):
                if _is_attn(l) and isinstance(l.keys, torch.Tensor) and l.keys.numel() > 0:
                    if getattr(l, "sliding_window", None) and l.keys.shape[-2] >= l.sliding_window:
                        raise NotImplementedError("Rewind über das Sliding Window hinaus ist nicht rekonstruierbar")
                    l.keys = l.keys[..., :length, :]
                    l.values = l.values[..., :length, :]
                if self.hybrid and _is_linear(l):
                    for a, val in self.snaps[length][i].items():
                        setattr(l, a, _clone_states(val))
            return c, length
        else:
            # Für Qwen3_5DynamicCache (flache Listen)
            if hasattr(cache, "key_cache"):
                for t in cache.key_cache:
                    if isinstance(t, torch.Tensor):
                        memo[id(t)] = t
            if hasattr(cache, "value_cache"):
                for t in cache.value_cache:
                    if isinstance(t, torch.Tensor):
                        memo[id(t)] = t
            c = copy.deepcopy(cache, memo)
            if hasattr(c, "key_cache") and hasattr(c, "value_cache"):
                for i in range(len(c.key_cache)):
                    k = c.key_cache[i]
                    v = c.value_cache[i]
                    if isinstance(k, torch.Tensor) and k.numel() > 0:
                        c.key_cache[i] = k[..., :length, :]
                    if isinstance(v, torch.Tensor) and v.numel() > 0:
                        c.value_cache[i] = v[..., :length, :]
            if self.hybrid and length in self.snaps:
                snap = self.snaps[length]
                for i, layer_snap in snap.items():
                    if "conv_states" in layer_snap and hasattr(c, "conv_states") and i < len(c.conv_states):
                        c.conv_states[i] = _clone_states(layer_snap["conv_states"])
                    if "recurrent_states" in layer_snap and hasattr(c, "recurrent_states") and i < len(c.recurrent_states):
                        c.recurrent_states[i] = _clone_states(layer_snap["recurrent_states"])
            return c, length
