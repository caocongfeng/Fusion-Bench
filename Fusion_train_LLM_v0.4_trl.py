#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import torch
from typing import Literal, Optional
from typing import Literal, Optional, List, Dict, Tuple

import wandb
import weave

from tqdm import tqdm

from datasets import load_dataset, Dataset

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextStreamer

import re
import math
import time

import pandas as pd
import json

from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model

import argparse

os.environ["FLASH_ATTENTION_FORCE_USE"] = "1"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多 GPU
    os.environ["PYTHONHASHSEED"] = str(seed)


# combination_meth=("N_T","T_N","Mixed")
CombinationMethod = Literal["N_T", "T_N", "Mix"]

EVAL_LIMIT = None   # None                  # "先跑一小段 limit=50"
BATCH_SIZE_GSM8K = 96            # 按显存调整；不做 argparse
BATCH_SIZE_OPENMATH = 24          # 按显存调整；不做 argparse
MAX_NEW_TOKENS_GSM8K = 1024       # 推理长度上限（GSM8K）
MAX_NEW_TOKENS_OPENMATH = 4096   # 推理长度上限（OpenMathReasoning）
TOKEN_STATS_SAMPLE = 1024         # 用于估算 token 分位数（越大越准，越慢）
PCTL_LIMIT = 95                  # 使用 P95 作为自动阈值
HARD_CAP_TOK = 8192              # “极长尾”硬上限；超过直接单独/过滤
FILTER_EXTREME = True           # True: 极长尾过滤；False: 极长尾单独 batch_size=1
NUM_TRAIN_EPOCES = 5              # num_train_epochs = 5


### Callback for evaluation at the end of each epoch
from transformers import TrainerCallback

class EvalEachEpochCallback(TrainerCallback):
    def __init__(self, reasoning_test, gsm8k_test, tokenizer, eval_limit, output_dir: Optional[str] = "./"):
        self.reasoning_test = reasoning_test
        self.gsm8k_test = gsm8k_test
        self.tokenizer = tokenizer
        self.eval_limit = eval_limit
        self.output_dir = output_dir  # NEW

    def on_epoch_end(self, args, state, control, **kwargs):
        print("\n=== EvalEachEpochCallback: Evaluation at the end of epoch ===")
        model = kwargs["model"]

        # 切到 eval 模式（原来的 after-training 逻辑）
        model.eval()
        try:
            model.gradient_checkpointing_disable()
        except Exception:
            pass
        model.config.use_cache = True

        # 取当前 epoch（state.epoch 可能是浮点）
        epoch = state.epoch
        if epoch is None:
            epoch_tag = f"step_{state.global_step}"
        else:
            epoch_tag = f"epoch_{int(round(epoch))}"

        print("="*20)
        print(f"[Callback] output_dir = {self.output_dir}")
        print("="*20)


        # ====== OpenMathReasoning ======
        result = evaluate_OpenMathReasoning(
            self.reasoning_test, model, self.tokenizer, limit=self.eval_limit, output_dir=self.output_dir+epoch_tag+"/"
        )
        print(f"[OpenMath][{epoch_tag}] acc/correct/total:",
              result["accuracy"], result["correct"], result["total"])
        print(f"[OpenMath][{epoch_tag}] ava_length(chars):", result["ava_length"])
        print(f"[OpenMath][{epoch_tag}] speed:", result["speed"])
        print(f"[OpenMath][{epoch_tag}] token_stats:", result["token_stats"])

        if wandb.run is not None:
            wandb.log(
                {
                    f"OpenMath/accuracy/{epoch_tag}": result["accuracy"],
                    f"OpenMath/correct/{epoch_tag}": result["correct"],
                    f"OpenMath/total/{epoch_tag}": result["total"],
                    f"OpenMath/ava_length/{epoch_tag}": result["ava_length"],
                    f"OpenMath/tok_per_s/{epoch_tag}": result["speed"]["tok_per_s"],
                    f"OpenMath/elapsed_s/{epoch_tag}": result["speed"]["elapsed_s"],
                    f"OpenMath/total_new_tokens/{epoch_tag}": result["speed"]["total_new_tokens"],
                    f"OpenMath/max_input_tokens/{epoch_tag}": result["token_stats"]["max_input_tokens"],
                    f"OpenMath/p95_tok/{epoch_tag}": result["token_stats"]["pctl_tok"],
                }
            )

        # os.makedirs(self.output_dir, exist_ok=True)
        # with open(os.path.join(self.output_dir, f"openmath_{epoch_tag}.json"), "w") as f:
        #     json.dump(result, f, ensure_ascii=False, indent=2)

        # ====== GSM8K ======
        result = evaluate_gsm8k(
            self.gsm8k_test, model, self.tokenizer, limit=self.eval_limit, output_dir=self.output_dir+epoch_tag+"/"
        )
        print(f"[GSM8K][{epoch_tag}] acc/correct/total:",
              result["accuracy"], result["correct"], result["total"])
        print(f"[GSM8K][{epoch_tag}] ava_length(chars):", result["ava_length"])
        print(f"[GSM8K][{epoch_tag}] speed:", result["speed"])
        print(f"[GSM8K][{epoch_tag}] token_stats:", result["token_stats"])

        if wandb.run is not None:
            wandb.log(
                {
                    f"GSM8K/accuracy/{epoch_tag}": result["accuracy"],
                    f"GSM8K/correct/{epoch_tag}": result["correct"],
                    f"GSM8K/total/{epoch_tag}": result["total"],
                    f"GSM8K/ava_length/{epoch_tag}": result["ava_length"],
                    f"GSM8K/tok_per_s/{epoch_tag}": result["speed"]["tok_per_s"],
                    f"GSM8K/elapsed_s/{epoch_tag}": result["speed"]["elapsed_s"],
                    f"GSM8K/total_new_tokens/{epoch_tag}": result["speed"]["total_new_tokens"],
                    f"GSM8K/max_input_tokens/{epoch_tag}": result["token_stats"]["max_input_tokens"],
                    f"GSM8K/p95_tok/{epoch_tag}": result["token_stats"]["pctl_tok"],
                }
            )

        # 训练继续
        model.train()
        return control


def generate_conversation(examples, prompt, type="reasoning"):
    if type == "reasoning":
        problems = examples["problem"]
        solutions = examples["generated_solution"]
        conversations = []
        for problem, solution in zip(problems, solutions):
            conversations.append(
                [
                    {"role": "user", "content": prompt.format(problem=problem)},
                    {"role": "assistant", "content": solution},
                ]
            )
    else:
        problems = examples["question"]
        solutions = examples["answer"]
        conversations = []
        for problem, solution in zip(problems, solutions):
            conversations.append(
                [
                    {"role": "user", "content": prompt.format(problem=problem)},
                    {"role": "assistant", "content": "\n<think>\n\n</think>\n\n" + solution},
                ]
            )
    return {"conversations": conversations}


def generate_non_reasoning_conversation(
    examples,
    prompt,
    type="non_reasoning",
):
    problems = examples["question"]
    solutions = examples["answer"]

    conversations = []

    for problem, solution in zip(problems, solutions):
        # Split solution
        parts = solution.split("\n####")
        if len(parts) != 2:
            raise ValueError(f"Unexpected solution format: {solution}")

        solu, ans = parts
        ans = ans.strip()

        conversations.append(
            [
                {
                    "role": "user",
                    "content": prompt.format(problem=problem),
                },
                {
                    "role": "assistant",
                    "content": (
                        "\n<think>\n\n</think>\n\n" + solu + "\n"
                        "**Final Answer**\n"
                        f"\\boxed{{{ans}}}."
                    ),
                },
            ]
        )

    return {"conversations": conversations}


def interleave_1_to_n(reasoning, non_reasoning, n):
    mixed = []
    r_len = len(reasoning)
    nr_len = len(non_reasoning)

    nr_idx = 0

    for r in reasoning:
        mixed.append(r)
        for _ in range(n):
            if nr_idx >= nr_len:
                break
            mixed.append(non_reasoning[nr_idx])
            nr_idx += 1

        if nr_idx >= nr_len:
            break

    return mixed


def extract_OpenMathReasoning_final_answer(text: str) -> Optional[str]:
    if not text:
        return None
    matches = re.findall(r"\\boxed\{\s*(.*?)\s*\}", text, flags=re.DOTALL)
    return matches[-1].strip() if matches else None


def extract_gsm8k_final_answer(text: str) -> Optional[str]:
    if text is None:
        return None
    m = re.search(r"\\boxed\{\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)\s*\}", text)
    if m:
        return m.group(1).replace(",", "")
    m = re.search(r"\*\*Final Answer\*\*[\s\n]*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if nums:
        return nums[-1].replace(",", "")
    return None


def extract_gsm8k_label_answer(example):
    ans = extract_gsm8k_final_answer(example["answer"])
    return ans.strip() if ans is not None else None


# =========================
# Prompt building (inference)
# =========================
def build_prompts(problems: List[str], tokenizer, enable_thinking: bool) -> List[str]:
    if enable_thinking:
        user_prompt = (
            "Please reason step by step but within 1000 words, and put your final answer within \\boxed{{}}.\n"
            "Problem: {problem}\n"
        )
        texts = []
        for p in problems:
            messages = [{"role": "user", "content": user_prompt.format(problem=p)}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            texts.append(text)
    else:
        user_prompt = (
            "Solve the problem, and put your final answer within \\boxed{{}}.\n"
            "Problem: {problem}\n"
        )

        texts = []
        for p in problems:
            messages = [{"role": "user", "content": user_prompt.format(problem=p)}]
            text = tokenizer.apply_chat_template(messages, tokenize=False,enable_thinking=enable_thinking, add_generation_prompt=True)
            texts.append(text)
    return texts


def token_len(text: str, tokenizer) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def estimate_token_thresholds(
    prompts: List[str],
    tokenizer,
    sample_n: int = TOKEN_STATS_SAMPLE,
    pctl: int = PCTL_LIMIT,
    hard_cap: int = HARD_CAP_TOK,
) -> Dict[str, int]:
    """
    基于 prompts 的 token 长度，抽样估计 Pxx 阈值，并给出最终建议 max_input_tokens。
    """
    n = len(prompts)
    if n == 0:
        return {"pctl": pctl, "pctl_tok": 0, "max_input_tokens": 0, "hard_cap": hard_cap}

    idx = np.random.choice(n, size=min(sample_n, n), replace=False)
    lens = np.array([token_len(prompts[i], tokenizer) for i in idx], dtype=np.int32)

    p_tok = int(np.percentile(lens, pctl))
    # 关键：max_input_tokens 取 min(Pxx, hard_cap)，防止长尾把阈值拉爆
    max_input_tokens = int(min(p_tok, hard_cap))

    return {
        "pctl": pctl,
        "pctl_tok": p_tok,
        "max_input_tokens": max_input_tokens,
        "hard_cap": hard_cap,
        "sample_n": int(len(lens)),
        "tok_min": int(lens.min()),
        "tok_mean": float(lens.mean()),
        "tok_p95": int(np.percentile(lens, 95)),
        "tok_max": int(lens.max()),
    }


def split_long_tail(
    prompts: List[str],
    tokenizer,
    max_input_tokens: int,
    hard_cap: int,
) -> Tuple[List[int], List[int], List[int]]:
    """
    返回三组索引：
      - normal_idx: <= max_input_tokens
      - long_idx: (max_input_tokens, hard_cap]
      - extreme_idx: > hard_cap
    注意：这里的长度计算在全量上会稍慢，但 limit=50 时完全可接受。
    """
    normal_idx, long_idx, extreme_idx = [], [], []
    for i, t in enumerate(prompts):
        L = token_len(t, tokenizer)
        if L <= max_input_tokens:
            normal_idx.append(i)
        elif L <= hard_cap:
            long_idx.append(i)
        else:
            extreme_idx.append(i)
    return normal_idx, long_idx, extreme_idx


def generate_batch(
    prompts: List[str],
    model,
    tokenizer,
    max_new_tokens: int,
    batch_size: int,
    max_input_tokens: int,
) -> Tuple[List[str], Dict[str, float]]:
    """
    批量 generate + tokens/s 统计。
    重要：
      - truncation_side='left' 保留末尾（更靠近 Problem）
      - padding_side='left' 更适合 generation
      - 显式 max_length=max_input_tokens 控制 prefill
    """
    model.eval()
    model.config.use_cache = True

    total_new_tokens = 0
    t0 = time.perf_counter()
    outputs: List[str] = []

    for s in tqdm(range(0, len(prompts), batch_size), desc="batch_generate", leave=False):
        batch_texts = prompts[s : s + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
        ).to(model.device)

        # 每个样本的 prompt 实际长度（考虑 truncation + padding）
        # prompt_lens = inputs["attention_mask"].sum(dim=1).tolist()
        input_len = inputs["input_ids"].shape[1]



        with torch.inference_mode():  # (2) inference_mode
            seq = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=False,  # (6) faster
            )

            

        for i in range(seq.size(0)):
            # gen_ids = seq[i, int(prompt_lens[i]) :]
            gen_ids = seq[i, input_len:]
            # total_new_tokens += int(gen_ids.numel())
            total_new_tokens += int((seq.size(1) - input_len))
            outputs.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

    t1 = time.perf_counter()
    elapsed = max(1e-9, t1 - t0)
    return outputs, {
        "elapsed_s": float(elapsed),
        "total_new_tokens": int(total_new_tokens),
        "tok_per_s": float(total_new_tokens / elapsed),
        "avg_new_tokens_per_sample": float(total_new_tokens / max(1, len(prompts))),
    }


def evaluate_gsm8k(dataset, model, tokenizer, limit=None, output_dir=""):
    save_json_path = "/gsm8k/test.jsonl"
    os.makedirs(os.path.dirname(output_dir+save_json_path), exist_ok=True)

    n = len(dataset) if limit is None else min(limit, len(dataset))
    problems = [dataset[i]["question"] for i in range(n)]
    gts = [extract_gsm8k_label_answer(dataset[i]) for i in range(n)]

    # Build prompts for token stats + batching
    prompts = build_prompts(problems, tokenizer, enable_thinking=False)

    # Token stats + threshold selection (auto)
    stats = estimate_token_thresholds(prompts, tokenizer)
    max_input_tokens = stats["max_input_tokens"]

    normal_idx, long_idx, extreme_idx = split_long_tail(
        prompts, tokenizer, max_input_tokens=max_input_tokens, hard_cap=HARD_CAP_TOK
    )

    print("[GSM8K] token stats:", stats)
    print(f"[GSM8K] normal={len(normal_idx)} long={len(long_idx)} extreme={len(extreme_idx)} "
          f"(max_input_tokens={max_input_tokens}, hard_cap={HARD_CAP_TOK})")

    # GSM8K 通常很短；这里仍按通用逻辑跑
    pred_texts = [None] * n
    speed_agg = {"elapsed_s": 0.0, "total_new_tokens": 0, "tok_per_s": 0.0}

    # 1) normal: batch
    if normal_idx:
        batch_prompts = [prompts[i] for i in normal_idx]
        outs, sp = generate_batch(
            batch_prompts,
            model,
            tokenizer,
            max_new_tokens=MAX_NEW_TOKENS_GSM8K,
            batch_size=BATCH_SIZE_GSM8K,
            max_input_tokens=max_input_tokens,
        )
        for k, i in enumerate(normal_idx):
            pred_texts[i] = outs[k]
        speed_agg["elapsed_s"] += sp["elapsed_s"]
        speed_agg["total_new_tokens"] += sp["total_new_tokens"]

    # 2) long: batch_size=1（避免 padding 拖累/oom）
    if long_idx:
        long_prompts = [prompts[i] for i in long_idx]
        outs, sp = generate_batch(
            long_prompts,
            model,
            tokenizer,
            max_new_tokens=MAX_NEW_TOKENS_GSM8K,
            batch_size=1,
            max_input_tokens=min(HARD_CAP_TOK, max_input_tokens),  # 仍然截断到阈值/上限
        )
        for k, i in enumerate(long_idx):
            pred_texts[i] = outs[k]
        speed_agg["elapsed_s"] += sp["elapsed_s"]
        speed_agg["total_new_tokens"] += sp["total_new_tokens"]

    # 3) extreme: 过滤或单独跑（仍然 batch_size=1 + 强截断 hard_cap）
    if extreme_idx:
        if FILTER_EXTREME:
            for i in extreme_idx:
                pred_texts[i] = ""  # filtered placeholder
        else:
            extreme_prompts = [prompts[i] for i in extreme_idx]
            outs, sp = generate_batch(
                extreme_prompts,
                model,
                tokenizer,
                max_new_tokens=MAX_NEW_TOKENS_GSM8K,
                batch_size=1,
                max_input_tokens=HARD_CAP_TOK,
            )
            for k, i in enumerate(extreme_idx):
                pred_texts[i] = outs[k]
            speed_agg["elapsed_s"] += sp["elapsed_s"]
            speed_agg["total_new_tokens"] += sp["total_new_tokens"]

    speed_agg["tok_per_s"] = float(speed_agg["total_new_tokens"] / max(1e-9, speed_agg["elapsed_s"]))
    print("[GSM8K] speed:", speed_agg)

    raw_outputs = pred_texts
    preds = [extract_gsm8k_final_answer(t) if t is not None else None for t in raw_outputs]
    lengths = [len(t) if t is not None else 0 for t in raw_outputs]

    # save jsonl
    # os.path.join(self.output_dir, f"openmath_{epoch_tag}.json")
    # os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
    with open(output_dir+save_json_path, "w") as f:
        for i in range(n):
            f.write(
                json.dumps(
                    {
                        "id": i,
                        "problem": problems[i],
                        "solution": raw_outputs[i],
                        "answer": gts[i],
                    }
                )
                + "\n"
            )
    print("Saving results to", save_json_path)

    correct = 0
    total = 0
    for p, gt in zip(preds, gts):
        if gt is None:
            continue
        total += 1
        if p == gt:
            correct += 1
    acc = correct / total if total > 0 else 0.0

    return {
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "preds": preds,
        "gts": gts,
        "raw_outputs": raw_outputs,
        "ava_length": float(np.mean(lengths)) if lengths else 0.0,
        "speed": speed_agg,
        "token_stats": stats,
    }


def evaluate_OpenMathReasoning(dataset, model, tokenizer, limit=None, output_dir=""):
    save_json_path = "/OpenMathReasoning/test.jsonl"
    os.makedirs(os.path.dirname(output_dir+save_json_path), exist_ok=True)
    # os.makedirs(self.output_dir, exist_ok=True)
    # with open(os.path.join(self.output_dir, f"openmath_{epoch_tag}.json"), "w") as f:
        #     json.dump(result, f, ensure_ascii=False, indent=2)

    n = len(dataset) if limit is None else min(limit, len(dataset))
    problems = [dataset[i]["problem"] for i in range(n)]
    gts = [dataset[i]["expected_answer"] for i in range(n)]

    prompts = build_prompts(problems, tokenizer, enable_thinking=True)

    stats = estimate_token_thresholds(prompts, tokenizer)
    max_input_tokens = stats["max_input_tokens"]

    normal_idx, long_idx, extreme_idx = split_long_tail(
        prompts, tokenizer, max_input_tokens=max_input_tokens, hard_cap=HARD_CAP_TOK
    )

    print("[OpenMath] token stats:", stats)
    print(f"[OpenMath] normal={len(normal_idx)} long={len(long_idx)} extreme={len(extreme_idx)} "
          f"(max_input_tokens={max_input_tokens}, hard_cap={HARD_CAP_TOK})")

    pred_texts = [None] * n
    speed_agg = {"elapsed_s": 0.0, "total_new_tokens": 0, "tok_per_s": 0.0}

    # normal: batch
    if normal_idx:
        batch_prompts = [prompts[i] for i in normal_idx]
        outs, sp = generate_batch(
            batch_prompts,
            model,
            tokenizer,
            max_new_tokens=MAX_NEW_TOKENS_OPENMATH,
            batch_size=BATCH_SIZE_OPENMATH,
            max_input_tokens=max_input_tokens,
        )
        for k, i in enumerate(normal_idx):
            pred_texts[i] = outs[k]
        speed_agg["elapsed_s"] += sp["elapsed_s"]
        speed_agg["total_new_tokens"] += sp["total_new_tokens"]

    # long: batch_size=1
    if long_idx:
        long_prompts = [prompts[i] for i in long_idx]
        outs, sp = generate_batch(
            long_prompts,
            model,
            tokenizer,
            max_new_tokens=MAX_NEW_TOKENS_OPENMATH,
            batch_size=1,
            max_input_tokens=min(HARD_CAP_TOK, max_input_tokens),
        )
        for k, i in enumerate(long_idx):
            pred_texts[i] = outs[k]
        speed_agg["elapsed_s"] += sp["elapsed_s"]
        speed_agg["total_new_tokens"] += sp["total_new_tokens"]

    # extreme: filter or single
    if extreme_idx:
        if FILTER_EXTREME:
            for i in extreme_idx:
                pred_texts[i] = ""
        else:
            extreme_prompts = [prompts[i] for i in extreme_idx]
            outs, sp = generate_batch(
                extreme_prompts,
                model,
                tokenizer,
                max_new_tokens=MAX_NEW_TOKENS_OPENMATH,
                batch_size=1,
                max_input_tokens=HARD_CAP_TOK,
            )
            for k, i in enumerate(extreme_idx):
                pred_texts[i] = outs[k]
            speed_agg["elapsed_s"] += sp["elapsed_s"]
            speed_agg["total_new_tokens"] += sp["total_new_tokens"]

    speed_agg["tok_per_s"] = float(speed_agg["total_new_tokens"] / max(1e-9, speed_agg["elapsed_s"]))
    print("[OpenMath] speed:", speed_agg)

    raw_outputs = pred_texts
    preds = [extract_OpenMathReasoning_final_answer(t) if t is not None else None for t in raw_outputs]
    lengths = [len(t) if t is not None else 0 for t in raw_outputs]

    # os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
    with open(output_dir+save_json_path, "w") as f:
        for i in range(n):
            f.write(
                json.dumps(
                    {
                        "id": i,
                        "problem": problems[i],
                        "solution": raw_outputs[i],
                        "answer": gts[i],
                    }
                )
                + "\n"
            )
    print("Saving results to", save_json_path)

    correct = 0
    total = 0
    for p, gt in zip(preds, gts):
        if gt is None:
            continue
        total += 1
        if p == gt:
            correct += 1
    acc = correct / total if total > 0 else 0.0

    return {
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "preds": preds,
        "gts": gts,
        "raw_outputs": raw_outputs,
        "ava_length": float(np.mean(lengths)) if lengths else 0.0,
        "speed": speed_agg,
        "token_stats": stats,
    }



def main(
    model_name="Qwen/Qwen3-4B",
    standard_index=1500,
    combination_method: CombinationMethod = "Mix",
    reasoning_chat_percentage=1,
    project_name="math",
):
    # ====== 原始配置：不改 ======
    seed = 42
    set_seed(seed)
    current_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    print("-" * 20)
    print("Current Time:", current_time)
    print("-" * 20)
    print("Combination method:", combination_method)
    print("-" * 20)

    # ====== 模型加载：替换为 Transformers + PEFT（其余不改） ======
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        device_map="auto",
        attn_implementation="sdpa",
    )



    # ====== 原始数据集与prompt：不改 ======
    reasoning_dataset = load_dataset("unsloth/OpenMathReasoning-mini", split="cot")

    reasoning_dataset_split = reasoning_dataset.train_test_split(test_size=0.1, seed=seed)

    reasoning_dataset_train = reasoning_dataset_split["train"]
    reasoning_dataset_test = reasoning_dataset_split["test"]
    reasoning_dataset_test=reasoning_dataset_test.select(range(int(standard_index*0.25)))
    print("reasoning_dataset_test",reasoning_dataset_test)

    EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN

    user_prompt_reasoning = (
        "Please reason step by step, and put your final answer within \\boxed{{}}.\n"
        "Problem: {problem}.\n"
    )

    user_prompt_non_reasoning = (
        "Solve the problem, and put your final answer within \\boxed{{}}.\n"
        "Problem: {problem}.\n"
    )

    # ====== 原始 conversations 生成：不改 ======
    conversations = generate_conversation(reasoning_dataset_train, prompt=user_prompt_reasoning, type="reasoning")
    reasoning_conversations = tokenizer.apply_chat_template(
        list(
            reasoning_dataset_train.map(
                lambda x: generate_conversation(x, prompt=user_prompt_reasoning, type="reasoning"),
                batched=True,
            )["conversations"]
        ),
        tokenize=False,
    )

    non_reasoning_dataset_train = load_dataset("openai/gsm8k", "main", split="train")
    non_reasoning_dataset_test = load_dataset("openai/gsm8k", "main", split="test")
    non_reasoning_dataset_test=non_reasoning_dataset_test.select(range(int(standard_index*0.25)))
    print("non_reasoning_dataset_test",non_reasoning_dataset_test)

    conversations = generate_conversation(
        non_reasoning_dataset_train,
        prompt=user_prompt_non_reasoning,
        type="non_reasoning",
    )

    non_reasoning_conversations = tokenizer.apply_chat_template(
        list(
            non_reasoning_dataset_train.map(
                lambda x: generate_non_reasoning_conversation(
                    x,
                    prompt=user_prompt_non_reasoning,
                    type="non_reasoning",
                ),
                batched=True,
            )["conversations"]
        ),
        tokenize=False,
    )
    print("===================data example===================")
    print("*"*20)
    print(non_reasoning_conversations[:5])
    print("*"*20)
    print(reasoning_conversations[:5])
    print("*"*20)
    print("===================")

    standard_reasoning_conversations = reasoning_conversations[:standard_index]
    standard_non_reasoning_conversations = non_reasoning_conversations[:standard_index]

    non_reasoning_subset = pd.Series(non_reasoning_conversations)
    standard_non_reasoning_subset = non_reasoning_subset.sample(
        int(len(standard_reasoning_conversations) * (1 / reasoning_chat_percentage)),
        random_state=seed,
        # replace=True,
    )
    print("-" * 20)
    print(len(standard_reasoning_conversations))
    print(len(standard_non_reasoning_subset))
    print(
        (len(standard_reasoning_conversations))
        / (len(standard_non_reasoning_subset) + len(standard_reasoning_conversations))
    )
    print("T/non-T", len(standard_reasoning_conversations) / len(standard_non_reasoning_subset))
    print("-" * 20)

    # ====== 原始 combination：不改 ======
    def combination(combination_meth: CombinationMethod):
        if combination_meth == "T_N":
            T_N_data = pd.concat(
                [
                    pd.Series(standard_reasoning_conversations),
                    pd.Series(standard_non_reasoning_subset),
                ],
                ignore_index=True,
            )
            T_N_data.name = "text"
            return Dataset.from_pandas(T_N_data.to_frame())

        elif combination_meth == "N_T":
            N_T_data = pd.concat(
                [
                    pd.Series(standard_non_reasoning_subset),
                    pd.Series(standard_reasoning_conversations),
                ],
                ignore_index=True,
            )
            N_T_data.name = "text"
            return Dataset.from_pandas(N_T_data.to_frame())

        elif combination_meth == "Mix":
            ratio = reasoning_chat_percentage  # 这里是 T / non_T

            if ratio <= 0:
                raise ValueError(f"reasoning_chat_percentage (T/non_T) must be > 0, got {ratio}")

            if ratio <= 1:
                n = max(1, int(round(1 / ratio)))
                print(f"Mix ratio(T/non_T)={ratio} => T:non_T = 1:{n}")

                mixed = interleave_1_to_n(
                    list(standard_reasoning_conversations),
                    list(standard_non_reasoning_subset),
                    n=n,
                )

            else:
                n = max(1, int(round(ratio)))
                print(f"Mix ratio(T/non_T)={ratio} => T:non_T = {n}:1")

                mixed = interleave_1_to_n(
                    list(standard_non_reasoning_subset),  # 主序列（“reasoning”形参位置）
                    list(standard_reasoning_conversations),  # 被插入序列（“non_reasoning”形参位置）
                    n=n,
                )

            s = pd.Series(mixed, name="text")
            return Dataset.from_pandas(s.to_frame(), preserve_index=False)

        else:
            raise ValueError("Invalid combination method. Choose from 'T_N', 'N_T', or 'Mix'.")

    combination_data = combination(combination_method)
    print("len(combination_data):", len(combination_data))
    print("combination_data[:2]:", combination_data[:20])

    # ====== 原始 wandb init：不改 ======
    run = wandb.init(
        project=project_name,
        entity="byfrfy",
        name=f"{model_name.split('/')[-1]}_{combination_method}_{reasoning_chat_percentage}_{current_time}",
        config={
            "model_name": model_name,
            "standard_index": standard_index,
            "combination_method": combination_method,
            "reasoning_chat_percentage": reasoning_chat_percentage,
            "project_name": project_name,

            "EVAL_LIMIT": EVAL_LIMIT,                                # "先跑一小段 limit=50"
            "BATCH_SIZE_GSM8K": BATCH_SIZE_GSM8K,                    # 按显存调整；不做 argparse
            "BATCH_SIZE_OPENMATH": BATCH_SIZE_OPENMATH,              # 按显存调整；不做 argparse
            "MAX_NEW_TOKENS_GSM8K": MAX_NEW_TOKENS_GSM8K,            # 推理长度上限（GSM8K）
            "MAX_NEW_TOKENS_OPENMATH": MAX_NEW_TOKENS_OPENMATH,      # 推理长度上限（OpenMathReasoning）
            "TOKEN_STATS_SAMPLE": TOKEN_STATS_SAMPLE,                # 用于估算 token 分位数（越大越准，越慢）
            "PCTL_LIMIT": PCTL_LIMIT,                                # 使用 P95 作为自动阈值
            "HARD_CAP_TOK": HARD_CAP_TOK,                            # “极长尾”硬上限；超过直接单独/过滤
            "FILTER_EXTREME": FILTER_EXTREME,                        # True: 极长尾过滤；False: 极长尾单独 batch_size=1
            "NUM_TRAIN_EPOCES": NUM_TRAIN_EPOCES,                    # num_train_epochs = 5

        }
    )
    # 

        # ====== after training: evaluate_gsm8K ======
    model.eval()
    model.gradient_checkpointing_disable() 
    model.config.use_cache = True

    # ====== before training: evaluate (limit=50 + batch + tokens/s) ======
    result = evaluate_OpenMathReasoning(reasoning_dataset_test, model, tokenizer, limit=EVAL_LIMIT, output_dir=output_dir)
    print("[OpenMath] acc/correct/total:", result["accuracy"], result["correct"], result["total"])
    print("[OpenMath] ava_length(chars):", result["ava_length"])
    print("[OpenMath] speed:", result["speed"])
    print("[OpenMath] token_stats:", result["token_stats"])

    if wandb.run is not None:
        wandb.log(
            {
                "ava_length_OpenMathReasoning_before": result["ava_length"],
                "accuracy_OpenMathReasoning_before": result["accuracy"],
                "correct_OpenMathReasoning_before": result["correct"],
                "total_OpenMathReasoning_before": result["total"],
                "openmath_tok_per_s_before": result["speed"]["tok_per_s"],
                "openmath_elapsed_s_before": result["speed"]["elapsed_s"],
                "openmath_total_new_tokens_before": result["speed"]["total_new_tokens"],
                "openmath_max_input_tokens_before": result["token_stats"]["max_input_tokens"],
                "openmath_p95_tok_before": result["token_stats"]["pctl_tok"],
            }
        )

    result = evaluate_gsm8k(non_reasoning_dataset_test, model, tokenizer, limit=EVAL_LIMIT, output_dir=output_dir)
    print("[GSM8K] acc/correct/total:", result["accuracy"], result["correct"], result["total"])
    print("[GSM8K] ava_length(chars):", result["ava_length"])
    print("[GSM8K] speed:", result["speed"])
    print("[GSM8K] token_stats:", result["token_stats"])

    if wandb.run is not None:
        wandb.log(
            {
                "ava_length_gsm8k_before": result["ava_length"],
                "accuracy_gsm8k_before": result["accuracy"],
                "correct_gsm8k_before": result["correct"],
                "total_gsm8k_before": result["total"],
                "gsm8k_tok_per_s_before": result["speed"]["tok_per_s"],
                "gsm8k_elapsed_s_before": result["speed"]["elapsed_s"],
                "gsm8k_total_new_tokens_before": result["speed"]["total_new_tokens"],
                "gsm8k_max_input_tokens_before": result["token_stats"]["max_input_tokens"],
                "gsm8k_p95_tok_before": result["token_stats"]["pctl_tok"],
            }
        )

    if wandb.run is not None:
        wandb.log(
            {
                "ava_length": result["ava_length"],
                "accuracy": result["accuracy"],
                "correct": result["correct"],
                "total": result["total"],
            }
        )
    
    # ====== statistics ======
    print("="* 20)
    print("max reasoning",np.max([len(c) for c in reasoning_dataset_train["generated_solution"]]))
    print("min reasoning",np.min([len(c) for c in reasoning_dataset_train["generated_solution"]]))
    print("mean reasoning",np.mean([len(c) for c in reasoning_dataset_train["generated_solution"]]))
    print("max reasoning_conversations",np.max([len(c) for c in reasoning_conversations]))
    print("min reasoning_conversations",np.min([len(c) for c in reasoning_conversations]))
    print("mean reasoning_conversations",np.mean([len(c) for c in reasoning_conversations]))
    print("="* 20)
    print("max non_reasoning_conversations",np.max([len(c) for c in non_reasoning_conversations]))
    print("min non_reasoning_conversations",np.min([len(c) for c in non_reasoning_conversations]))
    print("mean non_reasoning_conversations",np.mean([len(c) for c in non_reasoning_conversations]))
    print("="* 20)

    
    output_dir = (
        f"./adapter/{model_name.split('/')[-1].replace('-', '_')}"
        f"_comb_method_{combination_method}_{str(standard_index)}"
        f"_ratio_{str(reasoning_chat_percentage)}"
        f"_epochs_{NUM_TRAIN_EPOCES}"
        f"_{current_time}/"
    )

    # 训练常规设置
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=16,
        lora_alpha=64,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # ====== trainer： SFTTrainer  ======
    trainer = SFTTrainer(
        model=model,
        # tokenizer=tokenizer,
        train_dataset=combination_data,
        eval_dataset=None,  # Can set up evaluation!
        args=SFTConfig(
            dataset_text_field="text",
            output_dir=output_dir, # Will be set to adapter_dir later
            per_device_train_batch_size=1,
            gradient_accumulation_steps=6,  # Use GA to mimic batch size!
            warmup_steps=5,
            num_train_epochs=NUM_TRAIN_EPOCES,  # Set this for 1 full training run.
            # max_steps = 30,
            learning_rate=2e-5,  # Reduce to 2e-5 for long training runs
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="cosine",
            seed=seed,
            bf16=True, ##
            packing=False, ##packing=True
            max_length = 2048,  ##
            save_strategy="epoch", 
            report_to="wandb",  # Use TrackIO/WandB etc
        ),
        callbacks=[
            EvalEachEpochCallback(
                reasoning_test=reasoning_dataset_test,
                gsm8k_test=non_reasoning_dataset_test,
                tokenizer=tokenizer,
                eval_limit=EVAL_LIMIT,
                output_dir=output_dir,
                )
            ],
    )

    adapter_dir = (
        f"./adapter/{model_name.split('/')[-1].replace('-', '_')}"
        f"_comb_method_{combination_method}_{str(standard_index)}"
        f"_ratio_{str(reasoning_chat_percentage)}"
        f"_{current_time}_final/"
    )

    trainer_stats = trainer.train()
    trainer.save_model(adapter_dir)  # saves adapter_model + adapter_config (PEFT)
    trainer.tokenizer.save_pretrained(adapter_dir)

    # ====== after training: evaluate_gsm8K ======
    model.eval()
    model.gradient_checkpointing_disable() 
    model.config.use_cache = True

    # ====== after training: evaluate (limit=50 + batch + tokens/s) ======
    result = evaluate_OpenMathReasoning(reasoning_dataset_test, model, tokenizer, limit=EVAL_LIMIT, output_dir=output_dir)
    print("[OpenMath] acc/correct/total:", result["accuracy"], result["correct"], result["total"])
    print("[OpenMath] ava_length(chars):", result["ava_length"])
    print("[OpenMath] speed:", result["speed"])
    print("[OpenMath] token_stats:", result["token_stats"])

    if wandb.run is not None:
        wandb.log(
            {
                "ava_length_OpenMathReasoning_after": result["ava_length"],
                "accuracy_OpenMathReasoning_after": result["accuracy"],
                "correct_OpenMathReasoning_after": result["correct"],
                "total_OpenMathReasoning_after": result["total"],
                "openmath_tok_per_s": result["speed"]["tok_per_s"],
                "openmath_elapsed_s": result["speed"]["elapsed_s"],
                "openmath_total_new_tokens": result["speed"]["total_new_tokens"],
                "openmath_max_input_tokens": result["token_stats"]["max_input_tokens"],
                "openmath_p95_tok": result["token_stats"]["pctl_tok"],
            }
        )

    result = evaluate_gsm8k(non_reasoning_dataset_test, model, tokenizer, limit=EVAL_LIMIT, output_dir=output_dir)
    print("[GSM8K] acc/correct/total:", result["accuracy"], result["correct"], result["total"])
    print("[GSM8K] ava_length(chars):", result["ava_length"])
    print("[GSM8K] speed:", result["speed"])
    print("[GSM8K] token_stats:", result["token_stats"])

    if wandb.run is not None:
        wandb.log(
            {
                "ava_length_gsm8k_after": result["ava_length"],
                "accuracy_gsm8k_after": result["accuracy"],
                "correct_gsm8k_after": result["correct"],
                "total_gsm8k_after": result["total"],
                "gsm8k_tok_per_s": result["speed"]["tok_per_s"],
                "gsm8k_elapsed_s": result["speed"]["elapsed_s"],
                "gsm8k_total_new_tokens": result["speed"]["total_new_tokens"],
                "gsm8k_max_input_tokens": result["token_stats"]["max_input_tokens"],
                "gsm8k_p95_tok": result["token_stats"]["pctl_tok"],
            }
        )

    if wandb.run is not None:
        wandb.log(
            {
                "ava_length": result["ava_length"],
                "accuracy": result["accuracy"],
                "correct": result["correct"],
                "total": result["total"],
            }
        )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--standard_index", type=int, default=1500)
    parser.add_argument(
        "--combination_method",
        type=str,
        default="Mix",
        choices=["N_T", "T_N", "Mix"],
    )
    parser.add_argument(
        "--reasoning_chat_percentage",
        type=float,
        default=1,
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="math",
    )

    args = parser.parse_args()

    main(
        model_name=args.model_name,
        standard_index=args.standard_index,
        combination_method=args.combination_method,
        reasoning_chat_percentage=args.reasoning_chat_percentage,
        project_name=args.project_name,
    )
