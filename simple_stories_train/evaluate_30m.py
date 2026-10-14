"""Evaluate trained 30M models against the published SimpleStories-30M.

For each model (a local model_step_N.pt from train_llama.py, or a Hugging Face model id):
  - test loss over the dataset's test split, computed the same way as train_llama.py's
    validation loss (inputs = row[:-1], targets = row[1:]), and perplexity = exp(loss)
  - sampled completions for a fixed set of prompts

Every model is loaded into this repo's Llama class, and every weight must load: a missing
or unexpected weight is an error. (Llama.from_pretrained is not used. It loads with
strict=False, and its Hugging Face conversion does not match the published SimpleStories
models' weight names, so it silently leaves most of their weights at random values.)

Run from this directory (paths in the config are relative to it):
  python evaluate_30m.py --models ../out/<run>/checkpoints/model_step_N.pt SimpleStories/SimpleStories-30M
Writes <out_dir>/eval_results.json and <out_dir>/eval_report.md.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import yaml
from dataloaders import DatasetConfig, create_data_loader
from huggingface_hub import hf_hub_download
from models.llama import Llama
from models.model_configs import MODEL_CONFIGS
from safetensors.torch import load_file
from tokenizers import Tokenizer

PROMPTS = [
    "Once upon a time",
    "The little dog ran to the",
    "Mia looked at the sky and",
    "In a small village, there was a girl who",
    "The robot did not know how to",
]


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def hf_to_repo(hf: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename Hugging Face LlamaForCausalLM weights to this repo's Llama names.

    The repo keeps key and value projections in one matrix (kv_attn), split as K then V.
    Its rotary embedding matches Hugging Face's (tests/test_llama_implementation.py).
    """
    sd = {"transformer.wte.weight": hf["model.embed_tokens.weight"],
          "lm_head.weight": hf.get("lm_head.weight", hf["model.embed_tokens.weight"]),  # tied
          "transformer.rms_f.weight": hf["model.norm.weight"]}
    i = 0
    while f"model.layers.{i}.input_layernorm.weight" in hf:
        s, d = f"model.layers.{i}.", f"transformer.h.{i}."
        sd[d + "rms_1.weight"] = hf[s + "input_layernorm.weight"]
        sd[d + "rms_2.weight"] = hf[s + "post_attention_layernorm.weight"]
        sd[d + "attn.q_attn.weight"] = hf[s + "self_attn.q_proj.weight"]
        sd[d + "attn.kv_attn.weight"] = torch.cat(
            [hf[s + "self_attn.k_proj.weight"], hf[s + "self_attn.v_proj.weight"]], dim=0)
        sd[d + "attn.c_proj.weight"] = hf[s + "self_attn.o_proj.weight"]
        for name in ("gate_proj", "up_proj", "down_proj"):
            sd[d + f"mlp.{name}.weight"] = hf[s + f"mlp.{name}.weight"]
        i += 1
    used = 3 + 9 * i - ("lm_head.weight" not in hf)
    if used != len(hf):
        raise RuntimeError(f"Hugging Face checkpoint has {len(hf)} weights, converted {used}")
    return sd


def load_model(path_or_id: str, model_name: str, device: str) -> Llama:
    if os.path.exists(path_or_id):  # model_step_N.pt saved by train_llama.py
        sd = torch.load(path_or_id, weights_only=True, map_location="cpu")
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    else:  # Hugging Face model id
        sd = hf_to_repo(load_file(hf_hub_download(path_or_id, "model.safetensors")))
    # Rotary sin/cos tables are fixed buffers computed from the config, not learned weights.
    sd = {k: v for k, v in sd.items() if not k.endswith(("rotary_sin", "rotary_cos"))}
    model = Llama(MODEL_CONFIGS[model_name])
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.endswith(("rotary_sin", "rotary_cos"))]
    if missing or unexpected:
        raise RuntimeError(f"{path_or_id}: missing {missing[:5]}, unexpected {unexpected[:5]}")
    model.transformer.wte.weight = model.lm_head.weight  # keep input/output embeddings tied
    return model.to(device).eval()


@torch.no_grad()
def test_loss(model: Llama, dataset_config: DatasetConfig, batch_size: int, device: str,
              max_batches: int | None) -> tuple[float, int]:
    """Token-weighted mean loss over the test split (or its first max_batches batches)."""
    loader, _ = create_data_loader(dataset_config, batch_size=batch_size)
    total, count = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        rows = batch["input_ids"].to(torch.long).to(device)
        x, y = rows[:, :-1].contiguous(), rows[:, 1:].contiguous()  # forward() calls .view()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            _, loss = model(x, y, return_logits=False)
        total += loss.item() * y.numel()
        count += y.numel()
    return total / count, count


@torch.no_grad()
def completions(model: Llama, tok: Tokenizer, device: str, n_samples: int, temperature: float,
                max_new_tokens: int, seed: int) -> dict[str, list[str]]:
    eos = tok.token_to_id("[EOS]")
    torch.manual_seed(seed)
    results = {}
    for prompt in PROMPTS:
        ids = tok.encode(prompt).ids
        if ids and ids[-1] == eos:  # the tokenizer appends [EOS], which would end the story
            ids = ids[:-1]
        # Start with [EOS], as train_llama.py does: in training, every story follows an [EOS].
        x = torch.tensor([[eos] + ids] * n_samples, dtype=torch.long, device=device)
        out = model.generate(x, max_new_tokens, temperature=temperature, eos_token_id=eos)
        results[prompt] = [tok.decode(row.tolist(), skip_special_tokens=True) for row in out]
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True, help="local .pt files or Hugging Face ids")
    p.add_argument("--config", default="30M_config.yaml", help="training config (model and test data)")
    p.add_argument("--out_dir", default="../out/eval")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_batches", type=int, default=None, help="limit for quick tests")
    p.add_argument("--streaming", action="store_true", help="stream the test split (quick tests)")
    p.add_argument("--n_samples", type=int, default=2, help="completions per prompt per model")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = pick_device(args.device)
    cfg = yaml.safe_load(open(args.config))
    model_name = cfg["model_name"]
    data_cfg = DatasetConfig(**{**cfg["val_dataset_config"], "streaming": args.streaming})
    tok = Tokenizer.from_file(data_cfg.tokenizer_file_path)

    results = {"config": vars(args) | {"device": device, "model_name": model_name,
                                       "test_data": data_cfg.model_dump()}, "models": []}
    for name in args.models:
        t0 = time.time()
        print(f"evaluating {name} on {device}")
        model = load_model(name, model_name, device)
        n_params = sum(p.numel() for p in model.parameters())
        loss, n_tokens = test_loss(model, data_cfg, args.batch_size, device, args.max_batches)
        print(f"  params {n_params:,}  test loss {loss:.4f}  perplexity {math.exp(loss):.2f}"
              f"  ({n_tokens:,} tokens)")
        samples = completions(model, tok, device, args.n_samples, args.temperature,
                              args.max_new_tokens, args.seed)
        results["models"].append({"model": name, "params": n_params, "test_loss": loss,
                                  "perplexity": math.exp(loss), "test_tokens": n_tokens,
                                  "samples": samples, "seconds": round(time.time() - t0, 1)})
        del model

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval_results.json").write_text(json.dumps(results, indent=2))

    lines = ["# SimpleStories 30M evaluation", "",
             f"Test split of `{data_cfg.name}`, {results['models'][0]['test_tokens']:,} target tokens "
             f"in rows of {data_cfg.n_ctx}. Samples at temperature {args.temperature}.", "",
             "| Model | Parameters | Test loss | Perplexity |", "|---|---|---|---|"]
    lines += [f"| {m['model']} | {m['params'] / 1e6:.1f}M | {m['test_loss']:.4f} | {m['perplexity']:.2f} |"
              for m in results["models"]]
    for prompt in PROMPTS:
        lines += ["", f"## \"{prompt}\""]
        for m in results["models"]:
            for i, text in enumerate(m["samples"][prompt], 1):
                lines += ["", f"**{m['model']}** (sample {i}):", "", f"> {text.strip()}"]
    (out / "eval_report.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out / 'eval_results.json'} and {out / 'eval_report.md'}")


if __name__ == "__main__":
    main()
