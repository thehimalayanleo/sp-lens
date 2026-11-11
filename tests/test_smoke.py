"""End-to-end smoke test on a tiny random Llama, CPU only."""

from pathlib import Path

import pytest
import torch

from splens import SPLens, fit_lenses, load_model
from splens.fit import check_r_rules

MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    tmp = tmp_path_factory.mktemp("splens")
    tok = AutoTokenizer.from_pretrained(MODEL)
    config = LlamaConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64,
                         num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=4)
    LlamaForCausalLM(config).save_pretrained(tmp / "model")
    tok.save_pretrained(tmp / "model")
    model, tok = load_model(str(tmp / "model"), "cpu")
    text = tmp / "corpus.txt"
    text.write_text("\n\n".join(f"The number {i} is followed by {i + 1} and then {i + 2}. " * 4 for i in range(8)))
    return model, tok, tmp, str(text)


def test_r_rules_keep_logits(setup):
    model, tok, _, _ = setup
    ids = tok("Paris is the capital of France", return_tensors="pt").input_ids
    assert check_r_rules(model, ids) == 0.0


def test_fit_and_read(setup):
    model, tok, tmp, text = setup
    out = fit_lenses(model, tok, tmp / "lenses", prompts=2, length=16, corpus_file=text, log=lambda _: None)
    for rule in "jr":
        payload = torch.load(Path(out) / f"{rule}-lens" / "lens.pt", weights_only=False)
        d = model.config.hidden_size
        assert payload["J"].shape == (model.config.num_hidden_layers - 2, d, d)
    lens = SPLens(model, tok, out)
    assert set(lens.weights) == {"r", "j", "logit", "prev"}
    lens.calibrate(prompts=2, length=16, corpus_file=text)
    reading = lens.read("The number 3 is followed by", lenses=("logit", "j", "r"))
    assert len(reading.layers) == len(lens.labels)
    for layer in reading.layers:
        assert len(layer.sp) == 10
        assert len(set(layer.sp)) == 10
        assert set(layer.lenses) == {"logit", "j", "r"}
    assert reading.table()


def test_j_free(setup):
    model, tok, _, text = setup
    lens = SPLens(model, tok)
    lens.calibrate(prompts=2, length=16, corpus_file=text)
    reading = lens.read("The number 3 is followed by")
    assert all(len(layer.sp) == 10 for layer in reading.layers)
