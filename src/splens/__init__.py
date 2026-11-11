"""SP-Lens: read the concepts a language model holds in its middle layers."""

from .fit import fit_lenses
from .lens import J_FREE, JR_PREV, LayerReading, Reading, SPLens

__all__ = ["SPLens", "Reading", "LayerReading", "fit_lenses", "JR_PREV", "J_FREE", "load_model"]
__version__ = "0.1.0"


def load_model(model_id: str, device: str | None = None, dtype: str = "auto"):
    """Load a Hugging Face causal LM the way SP-Lens needs it (eager attention)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if dtype == "auto":
        dtype = "bfloat16" if device.startswith("cuda") else "float32"
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=getattr(torch, dtype), trust_remote_code=True, attn_implementation="eager"
    ).to(device).eval()
    return model, tok
