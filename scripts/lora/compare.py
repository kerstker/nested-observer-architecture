"""
Does the LoRA actually sound like her? Generate replies to the held-out conversations with and without
the adapter, next to what she really said, and count the things she does not do.

    .venv-lora/Scripts/python scripts/lora/compare.py [--n 12]
"""
import argparse
import json
import os
import re
import sys

import torch
from peft import PeftModel
sys.path.insert(0, os.path.dirname(__file__))
from common import base_config, load_tokenizer  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from build_dataset import DROP, edge  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
NO_THINK = "<think>\n\n</think>\n\n"

def generate(model, tok, messages, max_new_tokens=120):
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt += "\n</think>\n\n" if prompt.rstrip().endswith("<think>") else NO_THINK
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9,
                         pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    return re.sub(r"^.*?</think>\s*", "", text, flags=re.S).strip()

def faults(text):
    f = [name for name, rx in DROP if rx.search(text)] or ([] if text.strip() else ["空回复"])
    if edge(text) >= 0.4:
        f.append("语气太冲")
    if "<think>" in text or len(text) > 200:
        f.append("在长篇思考/长篇大论")
    return f

def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=os.path.join(ROOT, "models", "DeepSeek-R1-0528-Qwen3-8B"))
    p.add_argument("--adapter", default=os.path.join(ROOT, "models", "noa-lora"))
    p.add_argument("--data", default=os.path.join(ROOT, "data", "lora", "val.jsonl"))
    p.add_argument("--n", type=int, default=12)
    a = p.parse_args(argv)
    torch.manual_seed(0)
    tok = load_tokenizer(a.adapter)
    model = AutoModelForCausalLM.from_pretrained(a.base, config=base_config(a.base), dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(model, a.adapter).eval()
    with open(a.data, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f][:a.n]
    stats = {"lora": {"faults": 0, "chars": 0}, "base": {"faults": 0, "chars": 0}, "real": {"chars": 0}}
    for i, r in enumerate(rows):
        msgs, real = r["messages"][:-1], r["messages"][-1]["content"]
        with torch.no_grad():
            mine = generate(model, tok, msgs)
            with model.disable_adapter():
                base = generate(model, tok, msgs)
        print(f"\n#{i + 1} 对方：{msgs[-1]['content'][:60]}")
        print(f"  她真实说的：{real}")
        print(f"  LoRA      ：{mine}  {faults(mine) or ''}")
        print(f"  基座      ：{base[:200]}  {faults(base) or ''}")
        for k, t in (("lora", mine), ("base", base)):
            stats[k]["faults"] += bool(faults(t))
            stats[k]["chars"] += len(t)
        stats["real"]["chars"] += len(real)
    n = max(1, len(rows))
    print(f"\n有问题的回复：LoRA {stats['lora']['faults']}/{n}，基座 {stats['base']['faults']}/{n}")
    print(f"平均长度：她 {stats['real']['chars'] / n:.0f} 字，LoRA {stats['lora']['chars'] / n:.0f} 字，基座 {stats['base']['chars'] / n:.0f} 字")

if __name__ == "__main__":
    main(sys.argv[1:])
