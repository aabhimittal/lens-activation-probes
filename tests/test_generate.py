"""Grading and label-construction tests. No network, no model: the QA source and
the generation call are both stubbed, so what is under test is the labelling
logic itself."""
import json

import numpy as np
import pytest

from lens import generate as G


def test_normalize_answer_matches_squad_convention():
    assert G.normalize_answer("The  Beatles, and.") == "beatles and"
    assert G.normalize_answer("A Tale of Two Cities!") == "tale of two cities"
    assert G.normalize_answer("") == ""


def test_is_correct_is_lenient_about_sentence_answers():
    assert G.is_correct("The book was written by David Seville.", ["david seville"])
    assert G.is_correct("It's THE Beatles!", ["the beatles", "beatles"])
    assert not G.is_correct("I think it was Mark Twain", ["david seville"])
    assert not G.is_correct("anything", [""])  # empty alias must never match


def test_format_prompt_uses_chat_template_when_present():
    class TokChat:
        chat_template = "x"

        def apply_chat_template(self, msgs, tokenize, add_generation_prompt):
            return "<|im_start|>" + msgs[0]["content"]

    class TokPlain:
        chat_template = None

    assert G.format_prompt(TokChat(), "who?").startswith("<|im_start|>")
    assert "Question: who?" in G.format_prompt(TokPlain(), "who?")
    assert "Question: who?" in G.format_prompt(TokChat(), "who?", style="plain")


class StubBackend:
    """Answers correctly for questions containing 'easy', wrongly otherwise."""
    name = "stub"
    batch_size = 4
    tok = type("T", (), {"chat_template": None})()

    def __init__(self, answers):
        self._answers = answers


@pytest.fixture
def stub(monkeypatch):
    qs = [f"q{i} easy" if i % 4 == 0 else f"q{i} hard" for i in range(40)]
    aliases = [["gold"]] * 40
    monkeypatch.setattr(G, "load_triviaqa", lambda n, seed, split: (qs[:n], aliases[:n]))
    monkeypatch.setattr(G, "generate_answers",
                        lambda be, prompts, mnt, batch_size=None:
                        ["gold" if "easy" in p else "nonsense" for p in prompts])
    return StubBackend(None)


def test_labels_follow_grading_and_base_rate_is_reported(stub):
    ds, stats = G.build_verified_hallucination(stub, n=40, balance=False, verbose=False)
    assert len(ds) == 40
    assert abs(stats["raw_error_rate"] - 0.75) < 1e-9   # 1 in 4 is answerable
    assert ds.labels.sum() == 30                        # label 1 == model was wrong


def test_balancing_equalizes_classes_without_hiding_the_true_rate(stub):
    ds, stats = G.build_verified_hallucination(stub, n=40, balance=True, verbose=False)
    assert ds.labels.mean() == 0.5
    assert len(ds) == 20                                # 10 correct -> 10 + 10
    assert abs(stats["raw_error_rate"] - 0.75) < 1e-9   # the skew is still reported
    assert stats["n_generated"] == 40 and stats["n_kept"] == 20


def test_degenerate_label_set_raises_instead_of_producing_a_useless_probe(monkeypatch):
    monkeypatch.setattr(G, "load_triviaqa",
                        lambda n, seed, split: ([f"q{i}" for i in range(10)], [["gold"]] * 10))
    monkeypatch.setattr(G, "generate_answers",
                        lambda be, prompts, mnt, batch_size=None: ["nonsense"] * len(prompts))
    with pytest.raises(ValueError, match="one class is empty"):
        G.build_verified_hallucination(StubBackend(None), n=10, verbose=False)


def test_saved_jsonl_round_trips_through_the_normal_loader(stub, tmp_path):
    from lens.data import load_jsonl
    ds, stats = G.build_verified_hallucination(stub, n=40, verbose=False)
    p = G.save_labeled(ds, stats, tmp_path / "h.jsonl")
    back = load_jsonl(p)
    assert len(back) == len(ds)
    assert np.array_equal(back.labels, ds.labels)
    assert back.texts == ds.texts
    rec = json.loads(p.read_text().splitlines()[0])
    # The model's actual answer is kept so the grader can be audited by hand.
    assert "model_answer" in rec and "question" in rec
    assert json.loads(p.with_suffix(".meta.json").read_text())["raw_error_rate"] > 0


def test_label_drift_reports_agreement_against_the_fp16_label_set(monkeypatch):
    """Quantized generation is allowed to change the model's answers; what this
    must report is how many *labels* moved as a result."""
    from lens.quant import SPECS_BY_NAME

    prompts = [f"p{i}" for i in range(10)]
    aliases = [["gold"]] * 10

    calls = []

    def fake_gen(be, ps, mnt, batch_size=None, spec=None):
        calls.append(getattr(spec, "name", "fp16"))
        if spec is None:                       # fp16 reference: 6 of 10 correct
            return ["gold" if i < 6 else "wrong" for i in range(len(ps))]
        return ["gold" if i < 4 else "wrong" for i in range(len(ps))]

    monkeypatch.setattr(G, "generate_answers", fake_gen)
    be = type("B", (), {"restore_weights": lambda self, drop=False: None})()
    rows = G.label_drift(be, prompts, aliases, [SPECS_BY_NAME["w4-g128"]], verbose=False)

    assert calls == ["fp16", "w4-g128"]
    base, quant = rows[0], rows[1]
    assert base["spec"] == "fp16" and base["error_rate"] == 0.4 and base["label_agreement"] == 1.0
    assert quant["error_rate"] == 0.6          # two more questions now answered wrong
    assert quant["label_agreement"] == 0.8     # so 2 of 10 labels moved
    assert quant["answer_exact_match"] == 0.8


def test_label_drift_skips_the_baseline_spec(monkeypatch):
    from lens.quant import SPECS_BY_NAME
    monkeypatch.setattr(G, "generate_answers",
                        lambda be, ps, mnt, batch_size=None, spec=None: ["gold"] * len(ps))
    be = type("B", (), {"restore_weights": lambda self, drop=False: None})()
    rows = G.label_drift(be, ["p"] * 4, [["gold"]] * 4,
                         [SPECS_BY_NAME["fp16"], SPECS_BY_NAME["kv4"]], verbose=False)
    assert [r["spec"] for r in rows] == ["fp16", "kv4"]   # baseline appears once, not twice


def test_kappa_discounts_the_free_credit_a_broken_model_gets():
    """A model that answers everything wrong agrees with an 80%-wrong reference
    on 80% of items. Raw agreement calls that good; kappa must not."""
    ref = np.array([1] * 80 + [0] * 20)      # reference is wrong on 80
    dead = np.ones(100, dtype=int)           # quantized model is wrong on all
    assert (ref == dead).mean() == 0.8       # raw agreement looks respectable
    assert abs(G.cohens_kappa(ref, dead)) < 1e-9   # kappa sees through it

    assert G.cohens_kappa(ref, ref) == 1.0
    mixed = ref.copy()
    mixed[:10] = 0                           # 10 disagreements out of 100
    assert 0.5 < G.cohens_kappa(ref, mixed) < 1.0
