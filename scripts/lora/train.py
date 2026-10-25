"""
Train Noa's own LoRA on what she really said (data from build_dataset.py).

    .venv-lora/Scripts/python scripts/lora/train.py [--base models/DeepSeek-R1-0528-Qwen3-8B] [--out models/noa-lora]

The base is a reasoning model that thinks before it answers; in conversation she does not narrate
a chain of thought, so every target is written with an empty think block and the loss is taken on
her reply only. After training, the held-out conversations are scored under the base model and
under the LoRA (mean loss on her real replies): the adapter is saved together with that report,
and serve.py refuses an adapter that did not make her real replies more likely.
"""
import argparse
import json
import math
import os
import sys
import time

import torch
from peft import LoraConfig, get_peft_model
sys.path.insert(0, os.path.dirname(__file__))
from common import base_config, load_tokenizer  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
NO_THINK = "<think>\n\n</think>\n\n"

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def encode(tok, messages, max_len):
    """Prompt tokens are context (label -100); her reply (after an empty think block) is the target."""
    prompt = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    head = "\n</think>\n\n" if prompt.rstrip().endswith("<think>") else NO_THINK
    target = head + messages[-1]["content"] + tok.eos_token
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    t_ids = tok(target, add_special_tokens=False)["input_ids"]
    if len(t_ids) >= max_len:
        return None
    p_ids = p_ids[-(max_len - len(t_ids)):]
    return {"input_ids": p_ids + t_ids, "labels": [-100] * len(p_ids) + t_ids}

@torch.no_grad()
def mean_loss(model, batches):
    model.eval()
    total = n = 0.0
    for b in batches:
        ids = torch.tensor([b["input_ids"]], device=model.device)
        labels = torch.tensor([b["labels"]], device=model.device)
        out = model(input_ids=ids, labels=labels)
        k = sum(1 for x in b["labels"] if x != -100)
        total += out.loss.item() * k
        n += k
    return total / max(1, n)

def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=os.path.join(ROOT, "models", "DeepSeek-R1-0528-Qwen3-8B"))
    p.add_argument("--data", default=os.path.join(ROOT, "data", "lora"))
    p.add_argument("--out", default=os.path.join(ROOT, "models", "noa-lora"))
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--max-len", type=int, default=1536)
    a = p.parse_args(argv)

    torch.manual_seed(0)
    tok = load_tokenizer(a.base)
    train = [x for x in (encode(tok, e["messages"], a.max_len) for e in load_jsonl(os.path.join(a.data, "train.jsonl"))) if x]
    val = [x for x in (encode(tok, e["messages"], a.max_len) for e in load_jsonl(os.path.join(a.data, "val.jsonl"))) if x]
    print(f"train {len(train)}  val {len(val)}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(a.base, config=base_config(a.base), dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    base_val = mean_loss(model, val)
    print(f"base val loss {base_val:.4f}", flush=True)

    model = get_peft_model(model, LoraConfig(
        r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    steps = math.ceil(len(train) * a.epochs / a.accum)
    sched = get_cosine_schedule_with_warmup(opt, max(1, steps // 10), steps)
    gen = torch.Generator().manual_seed(0)
    history, started = [], time.time()
    for epoch in range(a.epochs):
        model.train()
        order = torch.randperm(len(train), generator=gen).tolist()
        running = 0.0
        for i, idx in enumerate(order):
            b = train[idx]
            out = model(input_ids=torch.tensor([b["input_ids"]], device=model.device),
                        labels=torch.tensor([b["labels"]], device=model.device))
            (out.loss / a.accum).backward()
            running += out.loss.item()
            if (i + 1) % a.accum == 0 or i == len(order) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
        v = mean_loss(model, val)
        history.append({"epoch": epoch + 1, "train_loss": running / len(order), "val_loss": v})
        print(f"epoch {epoch + 1}: train {running / len(order):.4f}  val {v:.4f}  ({time.time() - started:.0f}s)", flush=True)

    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    report = {"base": os.path.abspath(a.base), "train": len(train), "val": len(val), "base_val_loss": base_val,
              "lora_val_loss": history[-1]["val_loss"], "history": history,
              "better_than_base": history[-1]["val_loss"] < base_val, "trained_at": time.time()}
    with open(os.path.join(a.out, "noa_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main(sys.argv[1:])
