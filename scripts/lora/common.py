"""Shared by train.py / serve.py / compare.py."""
from transformers import AutoConfig

def base_config(base: str):
    """
    The base's config asks for YaRN RoPE scaling (x4, for 128k contexts) with an attn_factor that
    current transformers silently drops - which runs attention at the wrong temperature and breaks
    the model. Her conversations are far below Qwen3's native 32k, where no scaling is the
    recommended setting, so plain RoPE is used.
    """
    cfg = AutoConfig.from_pretrained(base)
    theta = (getattr(cfg, "rope_parameters", None) or {}).get("rope_theta") or getattr(cfg, "rope_theta", 1000000)
    cfg.rope_parameters = {"rope_type": "default", "rope_theta": theta}   # (rope_scaling is an alias of this)
    cfg.max_position_embeddings = 32768
    return cfg

def load_tokenizer(path: str):
    """
    The base declares tokenizer_class LlamaTokenizerFast; transformers 5 maps that to a
    sentencepiece-style tokenizer that ignores the byte-level BPE in tokenizer.json, so Chinese
    text encodes to nothing. Load tokenizer.json as it is.
    """
    from transformers import PreTrainedTokenizerFast
    return PreTrainedTokenizerFast.from_pretrained(path)
