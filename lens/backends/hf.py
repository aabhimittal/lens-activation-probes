"""HuggingFace transformers backend: real models, simulated serving configs.

Weight quantization is applied in place to every nn.Linear inside the decoder
blocks (embeddings, lm_head and norms are left in FP16 -- that is what real
INT4 pipelines do too). KV-cache quantization is applied through a Cache
subclass, so keys/values are quantized *before they are stored and attended
over*, which is the property that matters.

Memory note: restoring FP weights requires keeping originals. We keep them on
CPU, so peak host RAM is ~1x model size above the GPU copy. For big models,
prefer `reload_between_specs=True` and eat the load time instead.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..quant import QuantSpec
from .base import pool as pool_fn


def _quant_cache_cls():
    from transformers.cache_utils import DynamicCache

    class QuantKVCache(DynamicCache):
        """Fake-quantizes K/V at the moment they enter the cache."""
        spec: QuantSpec = None  # set by the backend before each forward

        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            if self.spec is not None and self.spec.kv_bits is not None:
                key_states = self.spec.quant_kv(key_states)
                value_states = self.spec.quant_kv(value_states)
            return super().update(key_states, value_states, layer_idx, cache_kwargs)

    return QuantKVCache


class HFBackend:
    def __init__(self, model_id: str, device: str = "auto", dtype: str = "auto",
                 max_len: int = 512, batch_size: int = 8, trust_remote_code: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.name = model_id
        self.max_len, self.batch_size = max_len, batch_size
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        td = {"auto": "auto", "float16": torch.float16, "bfloat16": torch.bfloat16,
              "float32": torch.float32}[dtype]
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        import transformers
        # transformers renamed torch_dtype -> dtype in v5; a wrong keyword is
        # silently swallowed into config kwargs, so pick it by version.
        dkw = ({"dtype": td} if int(transformers.__version__.split(".")[0]) >= 5
               else {"torch_dtype": td})
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=trust_remote_code, **dkw).to(device).eval()
        self.layers = self._find_layers()
        self.n_layers = len(self.layers)
        self.d_model = int(getattr(self.model.config, "hidden_size"))
        self._fp_backup: dict[str, "object"] = {}
        self._applied: Optional[str] = None

    def _find_layers(self):
        for attr in ("model.layers", "transformer.h", "model.decoder.layers", "gpt_neox.layers"):
            obj = self.model
            try:
                for part in attr.split("."):
                    obj = getattr(obj, part)
                return list(obj)
            except AttributeError:
                continue
        raise RuntimeError(f"could not locate decoder blocks on {type(self.model).__name__}")

    # ------------------------------------------------------- weight quant --
    def _linear_modules(self):
        import torch.nn as nn
        for i, blk in enumerate(self.layers):
            for name, mod in blk.named_modules():
                if isinstance(mod, nn.Linear):
                    yield f"{i}.{name}", mod

    def _ensure_backup(self):
        """Snapshot the FP weights once. They never change, so re-cloning them
        for every serving config is pure waste -- and on a real-model sweep the
        backup/restore traffic, not the forward pass, is what dominates."""
        if self._fp_backup:
            return
        with self.torch.no_grad():
            for key, mod in self._linear_modules():
                self._fp_backup[key] = mod.weight.detach().to("cpu", copy=True)

    def apply_weight_quant(self, spec: QuantSpec):
        """Quantize *from the pristine snapshot*, not from whatever is currently
        loaded, so switching between configs needs one pass instead of a restore
        pass followed by a re-clone."""
        if self._applied == spec.name:
            return
        if spec.weight_bits is None:
            self.restore_weights()
            self._applied = spec.name
            return
        self._ensure_backup()
        with self.torch.no_grad():
            for key, mod in self._linear_modules():
                # Quantize on the module's own device: on GPU this keeps the
                # arithmetic there rather than round-tripping through host.
                w = self._fp_backup[key].to(mod.weight.device, self.torch.float32)
                mod.weight.data.copy_(spec.quant_weight(w).to(mod.weight.dtype))
        self._applied = spec.name

    def restore_weights(self, drop: bool = False):
        """Put the FP weights back. The snapshot is kept by default (the next
        config will need it); pass drop=True to hand the host memory back when
        no more quantized configs are coming."""
        if not self._fp_backup:
            self._applied = None
            return
        with self.torch.no_grad():
            for key, mod in self._linear_modules():
                if key in self._fp_backup:
                    mod.weight.data.copy_(self._fp_backup[key].to(mod.weight.device,
                                                                 mod.weight.dtype))
        if drop:
            self._fp_backup.clear()
        self._applied = None

    # ------------------------------------------------------------- taps ----
    def activations(self, prompts: Sequence[str], layers: Sequence[int], spec: QuantSpec,
                    pooling: str = "last"):
        torch = self.torch
        layers = list(layers)
        self.apply_weight_quant(spec)
        cache_cls = _quant_cache_cls() if spec.kv_bits is not None else None
        buf: dict[int, list] = {l: [] for l in layers}
        grabbed: dict[int, "object"] = {}
        handles = []

        def mk_hook(l):
            def hook(_m, _i, out):
                grabbed[l] = (out[0] if isinstance(out, tuple) else out).detach()
            return hook

        def mk_pre_hook():
            def pre(_m, args, kwargs):
                if args:
                    return (spec.quant_act(args[0]),) + tuple(args[1:]), kwargs
                if "hidden_states" in kwargs:
                    kwargs["hidden_states"] = spec.quant_act(kwargs["hidden_states"])
                return args, kwargs
            return pre

        for l in layers:
            handles.append(self.layers[l].register_forward_hook(mk_hook(l)))
        if spec.act_bits is not None:
            for blk in self.layers:
                handles.append(blk.register_forward_pre_hook(mk_pre_hook(), with_kwargs=True))
        try:
            for i in range(0, len(prompts), self.batch_size):
                chunk = list(prompts[i : i + self.batch_size])
                enc = self.tok(chunk, return_tensors="pt", padding=True, truncation=True,
                               max_length=self.max_len).to(self.device)
                kw = {}
                if cache_cls is not None:
                    cache = cache_cls()
                    cache.spec = spec
                    kw["past_key_values"] = cache
                    kw["use_cache"] = True
                with torch.no_grad():
                    self.model(**enc, **kw)
                mask = enc["attention_mask"].bool().cpu().numpy()
                for l in layers:
                    H = grabbed[l].float().cpu().numpy()
                    for bi in range(H.shape[0]):
                        buf[l].append(pool_fn(H[bi][mask[bi]], pooling))
        finally:
            for h in handles:
                h.remove()
        return {l: np.asarray(v, dtype=np.float32) for l, v in buf.items()}
