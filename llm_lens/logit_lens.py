"""
Logit Lens — Project intermediate hidden states to vocabulary space.

At each layer, apply final LayerNorm + unembedding to see what token
the model would predict if decoding stopped there.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Optional

from .extractor import ActivationExtractor, ExtractionResult


@dataclass
class LogitLensResult:
    """Logit Lens projections for all layers."""

    tokens: list[str]
    num_layers: int
    top_tokens: list[list[tuple[str, float]]]  # per layer: [(token, prob), ...]
    entropy: Optional[np.ndarray] = None       # (num_layers,)

    def get_top1_sequence(self) -> list[str]:
        """Top-1 predicted token at each layer."""
        return [self.top_tokens[i][0][0] for i in range(self.num_layers)]

    def get_top1_probs(self) -> np.ndarray:
        """Top-1 probability at each layer."""
        return np.array([self.top_tokens[i][0][1] for i in range(self.num_layers)])

    def find_token_switches(self) -> list[tuple[int, str, str]]:
        """Find layers where the top-1 prediction changes.
        Returns: [(layer_idx, old_token, new_token), ...]"""
        seq = self.get_top1_sequence()
        switches = []
        for i in range(1, len(seq)):
            if seq[i] != seq[i - 1]:
                switches.append((i, seq[i - 1], seq[i]))
        return switches

    def summary(self) -> str:
        lines = ["Layer | Top-1           | Prob   | Entropy"]
        lines.append("-" * 50)
        for i in range(self.num_layers):
            tok, prob = self.top_tokens[i][0]
            ent = self.entropy[i] if self.entropy is not None else 0
            lines.append(f"  {i:3d} | {repr(tok):>15s} | {prob:.4f} | {ent:.2f}")
        return "\n".join(lines)


class LogitLens:
    """
    Apply Logit Lens analysis.

    Usage:
        lens = LogitLens(extractor)
        result = extractor.run("The capital of France is")
        lr = lens.decode(result)
        print(lr.summary())
        print(lr.find_token_switches())  # where does the prediction change?
    """

    def __init__(self, extractor: ActivationExtractor):
        self.tokenizer = extractor.tokenizer
        self.unembed = extractor.get_unembedding()
        self._ln = extractor.get_final_ln()

    def decode(self, result: ExtractionResult, token_idx: int = -1,
               top_k: int = 5) -> LogitLensResult:
        """Project every layer's hidden state through unembedding."""
        all_top = []
        entropies = []

        for i in range(result.num_layers):
            h = result.residual_stream[f"layer_{i}"][0, token_idx, :]
            h = self._apply_ln(h)
            logits = h @ self.unembed.T
            probs = F.softmax(logits, dim=-1)

            topk = torch.topk(probs, top_k)
            top_toks = [(self.tokenizer.decode(idx.item()), probs[idx].item())
                        for idx in topk.indices]
            all_top.append(top_toks)

            ent = -(probs * (probs + 1e-10).log()).sum().item()
            entropies.append(ent)

        return LogitLensResult(
            tokens=result.tokens, num_layers=result.num_layers,
            top_tokens=all_top, entropy=np.array(entropies),
        )

    def decode_all_tokens(self, result: ExtractionResult) -> list[list[str]]:
        """Top-1 prediction at every layer for every token position.
        Returns: evolution[layer][token_pos] = predicted_token_str."""
        evolution = []
        for i in range(result.num_layers):
            h = result.residual_stream[f"layer_{i}"][0]  # (seq_len, hidden)
            h = self._apply_ln_batch(h)
            logits = h @ self.unembed.T
            top1 = logits.argmax(dim=-1)
            tokens = [self.tokenizer.decode(t.item()) for t in top1]
            evolution.append(tokens)
        return evolution

    def _apply_ln(self, h):
        if self._ln is not None:
            h = self._ln(h.to(self._ln.weight.device).to(self._ln.weight.dtype))
            h = h.cpu().float()
        return h

    def _apply_ln_batch(self, h):
        if self._ln is not None:
            h = self._ln(h.to(self._ln.weight.device).to(self._ln.weight.dtype))
            h = h.cpu().float()
        return h
