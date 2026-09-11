#!/usr/bin/env python3
"""Bounded local Qwen inference or LoRA optimizer check; never saves model weights."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mode", choices=("inference", "lora"), default="inference")
    args = parser.parse_args()
    # CUDA JIT subprocesses (ninja) need the selected environment's executables.
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    if args.mode == "inference":
        from vllm import LLM, SamplingParams

        llm = LLM(model=str(args.model.resolve()), dtype="bfloat16", max_model_len=2048,
                  gpu_memory_utilization=0.35, enforce_eager=True, max_num_seqs=1)
        tokenizer = llm.get_tokenizer()
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Reply with the word ready."}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        output = llm.generate([prompt], SamplingParams(temperature=0, max_tokens=32))[0].outputs[0]
        if not output.token_ids:
            raise RuntimeError("vLLM returned no generated tokens")
        print(json.dumps({"mode": args.mode, "tokens": len(output.token_ids),
                          "text": output.text, "passed": True}))
    else:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation="flash_attention_2", device_map={"": 0},
        )
        model = get_peft_model(model, LoraConfig(
            r=8, lora_alpha=16, target_modules="all-linear", task_type="CAUSAL_LM",
        ))
        model.train()
        params = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=1e-4)
        before = [parameter.detach().clone() for parameter in params]
        inputs = tokenizer("Air quality forecast: PM2.5 decreases after the cold front.",
                           return_tensors="pt").to("cuda:0")
        loss = model(**inputs, labels=inputs["input_ids"]).loss
        if not torch.isfinite(loss).item():
            raise RuntimeError("non-finite loss")
        loss.backward()
        if not any(p.grad is not None for p in params) or not all(
            torch.isfinite(p.grad).all().item() for p in params if p.grad is not None
        ):
            raise RuntimeError("missing or non-finite gradients")
        optimizer.step()
        changed = sum(not torch.equal(old, new) for old, new in zip(before, params))
        if not changed:
            raise RuntimeError("optimizer did not update LoRA weights")
        print(json.dumps({"mode": args.mode, "loss": loss.item(),
                          "changed_parameter_tensors": changed, "passed": True}))


if __name__ == "__main__":
    main()
