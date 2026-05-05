"""
Activation Extractor — Hook-based capture of internal layer outputs.

Supports: Qwen2, Llama, Mistral, Phi, GPT-NeoX and auto-detection for others.
Captures residual stream (post-layer hidden states) for all layers in one forward pass.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExtractionResult:
    """All captured activations from a single forward pass."""

    input_text: str
    input_ids: torch.Tensor                    # (1, seq_len)
    tokens: list[str]                          # decoded token strings
    residual_stream: dict[str, torch.Tensor]   # "layer_i" -> (1, seq_len, hidden_dim)
    embedding_output: Optional[torch.Tensor] = None

    # Per-head pre-projection outputs, populated only when extractor was created
    # with capture_heads=True. Each value is the o_proj input reshaped:
    #   (1, seq_len, num_heads, head_dim)
    head_outputs: dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def num_layers(self) -> int:
        return len(self.residual_stream)

    @property
    def seq_len(self) -> int:
        return self.input_ids.shape[1]

    @property
    def hidden_dim(self) -> int:
        if self.residual_stream:
            return next(iter(self.residual_stream.values())).shape[-1]
        return 0

    @property
    def num_heads(self) -> int:
        if self.head_outputs:
            return next(iter(self.head_outputs.values())).shape[2]
        return 0

    @property
    def head_dim(self) -> int:
        if self.head_outputs:
            return next(iter(self.head_outputs.values())).shape[3]
        return 0

    def get_residuals(self, token_idx: int = -1) -> torch.Tensor:
        """Stack residual stream for one token across all layers -> (num_layers, hidden_dim)."""
        vectors = []
        for i in range(self.num_layers):
            h = self.residual_stream[f"layer_{i}"][0, token_idx, :]
            vectors.append(h)
        return torch.stack(vectors)

    def get_all_token_residuals(self) -> torch.Tensor:
        """All tokens, all layers -> (num_layers, seq_len, hidden_dim)."""
        layers = []
        for i in range(self.num_layers):
            layers.append(self.residual_stream[f"layer_{i}"][0])
        return torch.stack(layers)

    def get_head_outputs(self, token_idx: int = -1) -> torch.Tensor:
        """Per-head pre-projection vectors for one token across all layers.

        Returns: (num_layers, num_heads, head_dim)
        Raises ValueError if head outputs were not captured.
        """
        if not self.head_outputs:
            raise ValueError(
                "No head outputs captured. Re-create ActivationExtractor with "
                "capture_heads=True.")
        layers = []
        for i in range(self.num_layers):
            h = self.head_outputs[f"layer_{i}"][0, token_idx]  # (num_heads, head_dim)
            layers.append(h)
        return torch.stack(layers)

    def get_all_token_head_outputs(self) -> torch.Tensor:
        """All tokens, all layers, per-head -> (num_layers, seq_len, num_heads, head_dim)."""
        if not self.head_outputs:
            raise ValueError(
                "No head outputs captured. Re-create ActivationExtractor with "
                "capture_heads=True.")
        layers = []
        for i in range(self.num_layers):
            layers.append(self.head_outputs[f"layer_{i}"][0])
        return torch.stack(layers)


class ActivationExtractor:
    """
    Wraps a HuggingFace causal LM, registers hooks to capture residual stream.

    Usage:
        ext = ActivationExtractor("Qwen/Qwen2.5-3B")
        result = ext.run("Hello world")
        print(result.num_layers, result.hidden_dim)
    """

    # Per-arch paths: (layers_attr, embed_attr, attn_oproj_relpath_within_layer)
    ARCH_MAP = {
        "Qwen2ForCausalLM":    ("model.layers",    "model.embed_tokens",   "self_attn.o_proj"),
        "Qwen3ForCausalLM":    ("model.layers",    "model.embed_tokens",   "self_attn.o_proj"),
        "LlamaForCausalLM":    ("model.layers",    "model.embed_tokens",   "self_attn.o_proj"),
        "MistralForCausalLM":  ("model.layers",    "model.embed_tokens",   "self_attn.o_proj"),
        "Phi3ForCausalLM":     ("model.layers",    "model.embed_tokens",   "self_attn.o_proj"),
        "GPTNeoXForCausalLM":  ("gpt_neox.layers", "gpt_neox.embed_in",    "attention.dense"),
    }

    def __init__(self, model_name_or_path: str, device: str = "auto",
                 dtype: torch.dtype = torch.float32, capture_heads: bool = False):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name_or_path
        self.dtype = dtype
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.capture_heads = capture_heads

        print(f"Loading {model_name_or_path} on {self.device} ({dtype})...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        except Exception as e:
            print(f"  [WARN] Fast tokenizer failed ({e.__class__.__name__}), falling back to slow tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, trust_remote_code=True, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=dtype,
            device_map=self.device if self.device != "cpu" else None,
            trust_remote_code=True,
        )
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        self.model.eval()

        # Detect architecture
        arch = type(self.model).__name__
        if arch in self.ARCH_MAP:
            info = self.ARCH_MAP[arch]
            layers_attr, embed_attr = info[0], info[1]
            attn_oproj_attr = info[2] if len(info) > 2 else "self_attn.o_proj"
        else:
            layers_attr, embed_attr = self._auto_detect()
            attn_oproj_attr = "self_attn.o_proj"  # safe default for most modern decoders

        self._layers = self._nested_attr(self.model, layers_attr)
        self._embed = self._nested_attr(self.model, embed_attr)
        self._lm_head = self.model.lm_head
        self._attn_oproj_attr = attn_oproj_attr
        self.num_layers = len(self._layers)
        self.hidden_dim = self.model.config.hidden_size
        self.param_count = sum(p.numel() for p in self.model.parameters())

        # Detect attention head config (only meaningful when capture_heads=True)
        cfg = self.model.config
        self.num_heads = (getattr(cfg, "num_attention_heads", None)
                          or getattr(cfg, "num_heads", None))
        self.head_dim = getattr(cfg, "head_dim", None)
        if self.head_dim is None and self.num_heads:
            self.head_dim = self.hidden_dim // self.num_heads

        head_msg = (f", heads={self.num_heads}×{self.head_dim}"
                    if self.capture_heads else "")
        print(f"Ready: {self.num_layers} layers, dim={self.hidden_dim}{head_msg}, "
              f"params={self.param_count / 1e9:.2f}B")

        self._hooks = []
        self._result = None

    def run(self, text: str) -> ExtractionResult:
        """Forward pass with hook capture -> ExtractionResult."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        tokens = [self.tokenizer.decode(t) for t in input_ids[0]]

        self._result = ExtractionResult(
            input_text=text, input_ids=input_ids.cpu(), tokens=tokens,
            residual_stream={},
        )
        self._register_hooks()

        with torch.no_grad():
            self.model(**inputs)

        self._remove_hooks()
        result = self._result
        self._result = None
        return result

    def get_unembedding(self) -> torch.Tensor:
        return self._lm_head.weight.detach().cpu().float()

    def get_final_ln(self) -> Optional[nn.Module]:
        for path in ["model.norm", "gpt_neox.final_layer_norm"]:
            try:
                return self._nested_attr(self.model, path)
            except AttributeError:
                continue
        return None

    def _register_hooks(self):
        self._remove_hooks()

        def embed_hook(m, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            self._result.embedding_output = h.detach().cpu().float()
        self._hooks.append(self._embed.register_forward_hook(embed_hook))

        for i, layer in enumerate(self._layers):
            name = f"layer_{i}"

            def make_residual_hook(n):
                def hook(m, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    self._result.residual_stream[n] = h.detach().cpu().float()
                return hook
            self._hooks.append(layer.register_forward_hook(make_residual_hook(name)))

            # Per-head capture: hook the o_proj forward_pre to grab the
            # concatenated head output (bsz, seq, num_heads * head_dim).
            if self.capture_heads:
                try:
                    o_proj = self._nested_attr(layer, self._attn_oproj_attr)
                except AttributeError:
                    continue

                def make_head_hook(n):
                    def hook(m, inp):
                        if not inp:
                            return
                        x = inp[0]
                        if not torch.is_tensor(x) or x.dim() != 3:
                            return
                        bsz, seq_len, hidden = x.shape
                        nh = self.num_heads or 1
                        if nh <= 0 or hidden % nh != 0:
                            return
                        hd = hidden // nh
                        per_head = x.view(bsz, seq_len, nh, hd)
                        self._result.head_outputs[n] = per_head.detach().cpu().float()
                    return hook
                self._hooks.append(o_proj.register_forward_pre_hook(make_head_hook(name)))

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    @staticmethod
    def _nested_attr(obj, path):
        for part in path.split("."):
            obj = getattr(obj, part)
        return obj

    def _auto_detect(self):
        for lp in ["model.layers", "transformer.h", "gpt_neox.layers"]:
            try:
                layers = self._nested_attr(self.model, lp)
                if isinstance(layers, nn.ModuleList) and len(layers) > 0:
                    for ep in ["model.embed_tokens", "transformer.wte", "gpt_neox.embed_in"]:
                        try:
                            self._nested_attr(self.model, ep)
                            return lp, ep
                        except AttributeError:
                            continue
            except AttributeError:
                continue
        raise RuntimeError(f"Cannot detect architecture for {type(self.model).__name__}")

    def __repr__(self):
        return f"ActivationExtractor({self.model_name}, layers={self.num_layers})"
