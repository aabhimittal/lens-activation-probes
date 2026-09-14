"""Verified hallucination labels: let the model answer, then grade it.

The bundled `hallucination` task labels *prompts that invite fabrication*. That
is a proxy, and the weakest label in the repo. This module replaces it with the
label the question actually deserves: run the model on a real closed-book QA
set, grade its greedy answer against gold aliases, and label an example 1 when
the model **got it wrong**.

That changes what the probe is. It is no longer "does this prompt look risky";
it is "will *this model* fabricate on this prompt", read off the residual stream
at the last prompt token, before a single answer token is generated. That is the
claim the hallucination-probe literature makes, and it is the one worth testing
under quantization.

Two consequences worth being explicit about:

* The labels are **model-specific**. A label set built against Qwen2.5-0.5B says
  nothing about another model, and re-running against a quantized stack would
  change the labels themselves -- which is why labels are always generated once,
  on the FP16 reference stack, and then held fixed while the serving config
  varies underneath. Otherwise the target moves with the thing being measured.
* Small models are wrong most of the time, so the raw base rate is badly skewed.
  `balance=True` subsamples the majority class and the true base rate is
  reported alongside, rather than quietly dropped.
"""
from __future__ import annotations

import json
import re
import string
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .data import Dataset

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize_answer(s: str) -> str:
    """SQuAD-style normalization: lowercase, drop punctuation and articles,
    collapse whitespace. Deliberately the standard one so numbers are
    comparable to published QA results."""
    s = s.lower().translate(_PUNCT)
    return " ".join(_ARTICLES.sub(" ", s).split())


def is_correct(output: str, aliases: Sequence[str]) -> bool:
    """Substring match against any gold alias.

    Lenient on purpose: a small model answers in a sentence ("The book was
    written by ..."), so exact match would score almost everything wrong and the
    label would degenerate into "did the model produce a bare noun phrase".
    Lenient matching over-credits (a long answer containing the gold string by
    luck), which biases *against* finding probe signal -- the safe direction.
    """
    o = normalize_answer(output)
    return any(a and normalize_answer(a) in o for a in aliases)


def format_prompt(tok, question: str, style: str = "auto") -> str:
    """Format one question the way the model is actually served."""
    if style == "plain" or (style == "auto" and getattr(tok, "chat_template", None) is None):
        return f"Answer the question.\nQuestion: {question}\nAnswer:"
    msgs = [{"role": "user", "content": f"Answer the question in a few words.\n{question}"}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def generate_answers(backend, prompts: Sequence[str], max_new_tokens: int = 16,
                     batch_size: Optional[int] = None, spec=None) -> list[str]:
    """Greedy decode. Left padding, because a decoder-only model batched with
    right padding generates from pad tokens and returns garbage.

    Pass `spec` to generate from a *quantized* stack instead of FP16. That is
    not how training labels are made (see the module docstring) -- it is how
    label drift is measured: the same questions, answered by the serving config
    that will actually be deployed.
    """
    import torch

    bs = batch_size or backend.batch_size
    tok, model = backend.tok, backend.model
    cache_cls = None
    if spec is not None:
        backend.apply_weight_quant(spec)
        if spec.kv_bits is not None:
            from .backends.hf import _quant_cache_cls
            cache_cls = _quant_cache_cls()
    prev_side = tok.padding_side
    tok.padding_side = "left"
    out: list[str] = []
    try:
        for i in range(0, len(prompts), bs):
            chunk = list(prompts[i : i + bs])
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                      max_length=backend.max_len).to(backend.device)
            kw = {}
            if cache_cls is not None:
                cache = cache_cls()
                cache.spec = spec
                kw["past_key_values"] = cache
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                     pad_token_id=tok.pad_token_id, **kw)
            new = gen[:, enc["input_ids"].shape[1]:]
            out.extend(tok.batch_decode(new, skip_special_tokens=True))
    finally:
        tok.padding_side = prev_side
    return out


def load_triviaqa(n: int, seed: int = 0, split: str = "validation"):
    """-> (questions, alias lists). Needs `datasets`; TriviaQA ships normalized
    answer aliases, which is what makes lenient grading defensible."""
    from datasets import load_dataset

    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split=split)
    idx = np.random.default_rng(seed).permutation(len(ds))[: n]
    qs, als = [], []
    for i in idx:
        r = ds[int(i)]
        aliases = r["answer"].get("normalized_aliases") or [r["answer"].get("value", "")]
        if r["question"] and aliases:
            qs.append(r["question"])
            als.append(list(aliases))
    return qs, als


def build_verified_hallucination(backend, n: int = 800, seed: int = 0,
                                 max_new_tokens: int = 16, balance: bool = True,
                                 prompt_style: str = "auto", split: str = "validation",
                                 verbose: bool = True) -> tuple[Dataset, dict]:
    """Label 1 = the model's greedy answer did NOT contain a gold alias."""
    questions, aliases = load_triviaqa(n, seed, split)
    prompts = [format_prompt(backend.tok, q, prompt_style) for q in questions]
    answers = generate_answers(backend, prompts, max_new_tokens)
    labels = np.array([0 if is_correct(a, al) else 1 for a, al in zip(answers, aliases)])
    stats = {"n_generated": len(labels), "raw_error_rate": float(labels.mean()),
             "model": backend.name, "source": f"trivia_qa/rc.nocontext/{split}",
             "max_new_tokens": max_new_tokens, "balanced": bool(balance)}
    keep = np.arange(len(labels))
    if balance:
        rng = np.random.default_rng(seed)
        pos, neg = np.where(labels == 1)[0], np.where(labels == 0)[0]
        k = min(len(pos), len(neg))
        if k == 0:
            raise ValueError(f"one class is empty (error rate {labels.mean():.2f}); "
                             "use a stronger model or more questions")
        keep = np.concatenate([rng.choice(pos, k, replace=False),
                               rng.choice(neg, k, replace=False)])
        rng.shuffle(keep)
    stats["n_kept"] = int(keep.size)
    if verbose:
        print(f"[label] {stats['n_generated']} questions, raw error rate "
              f"{stats['raw_error_rate']:.2f} -> kept {stats['n_kept']} "
              f"(balanced={balance})")
    ds = Dataset([prompts[i] for i in keep], labels[keep], "hallucination_verified",
                 np.asarray(keep))
    ds.answers = [answers[i] for i in keep]      # kept for auditing the grader
    ds.questions = [questions[i] for i in keep]
    return ds, stats


def cohens_kappa(y_ref: np.ndarray, y_q: np.ndarray) -> float:
    """Chance-corrected agreement.

    Raw agreement is untrustworthy here because the base rate is extreme. A
    quantized model that gets *everything* wrong automatically agrees with the
    FP16 labels on every question FP16 also got wrong -- which, at an 80% error
    rate, is 80% agreement for a model that has been destroyed. Kappa subtracts
    exactly that free credit.
    """
    y_ref = np.asarray(y_ref)
    y_q = np.asarray(y_q)
    po = float((y_ref == y_q).mean())
    p1, q1 = float(y_ref.mean()), float(y_q.mean())
    pe = p1 * q1 + (1 - p1) * (1 - q1)
    return float((po - pe) / (1 - pe)) if pe < 1.0 else float("nan")


def label_drift(backend, prompts: Sequence[str], aliases: Sequence[Sequence[str]],
                specs, max_new_tokens: int = 16, verbose: bool = True) -> list[dict]:
    """How much does quantization change *which questions the model gets wrong*?

    This is the question the sweep deliberately does not ask. The sweep freezes
    labels at FP16 so that ΔAUROC measures probe degradation and nothing else.
    But in production the quantized model is the one generating, so a probe can
    be perfectly robust and still be wrong about the deployed system, simply
    because the deployed system fails on a different set of inputs.

    Returns per-spec error rate and agreement with the FP16 label set.
    """
    ref = generate_answers(backend, prompts, max_new_tokens)
    y_ref = np.array([0 if is_correct(a, al) else 1 for a, al in zip(ref, aliases)])
    out = [{"spec": "fp16", "error_rate": float(y_ref.mean()), "label_agreement": 1.0,
            "kappa": 1.0, "answer_exact_match": 1.0, "n": int(y_ref.size),
            "kept_correct": int((y_ref == 0).sum()), "was_correct": int((y_ref == 0).sum())}]
    for spec in specs:
        if spec.is_baseline:
            continue
        got = generate_answers(backend, prompts, max_new_tokens, spec=spec)
        y = np.array([0 if is_correct(a, al) else 1 for a, al in zip(got, aliases)])
        rec = {"spec": spec.name, "error_rate": float(y.mean()),
               "label_agreement": float((y == y_ref).mean()),
               "kappa": cohens_kappa(y_ref, y),
               # Of the questions FP16 answered correctly, how many survive?
               # This is the number an operator actually feels.
               "was_correct": int((y_ref == 0).sum()),
               "kept_correct": int(((y_ref == 0) & (y == 0)).sum()),
               "answer_exact_match": float(np.mean([a.strip() == b.strip()
                                                    for a, b in zip(got, ref)])),
               "n": int(y.size)}
        out.append(rec)
        if verbose:
            print(f"[drift] {spec.name}: error {rec['error_rate']:.3f} "
                  f"(fp16 {y_ref.mean():.3f}), agreement "
                  f"{rec['label_agreement']:.3f} (kappa {rec['kappa']:.3f}), "
                  f"kept {rec['kept_correct']}/{rec['was_correct']} correct answers, "
                  f"identical answers {rec['answer_exact_match']:.3f}")
    backend.restore_weights(drop=True)
    return out


def save_labeled(ds: Dataset, stats: dict, path) -> Path:
    """JSONL that `lens.data.load_jsonl` can read straight back."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for i, (t, y, g) in enumerate(zip(ds.texts, ds.labels, ds.groups)):
            rec = {"text": t, "label": int(y), "group": int(g)}
            if hasattr(ds, "questions"):
                rec["question"] = ds.questions[i]
                rec["model_answer"] = ds.answers[i]
            f.write(json.dumps(rec) + "\n")
    p.with_suffix(".meta.json").write_text(json.dumps(stats, indent=2))
    return p
