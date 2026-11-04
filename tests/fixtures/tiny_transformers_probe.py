"""Offline CPU integration probe with random tiny weights, never an LLM quality benchmark."""
import argparse
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from vibeforge.contracts.precision import normalize_precision, precision_fields
from vibeforge.engine.process import EngineProcess
from vibeforge.evaluation.correctness import evaluate_correctness
from vibeforge.evaluation.reference import capture_reference
from vibeforge.storage.atomic import write_json
from vibeforge.storage.paths import Paths
from vibeforge.versioning.candidate import create_workspace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    project, output = Path(args.project), Path(args.output)
    paths = Paths.create(project, output / "var")
    torch.set_num_threads(1)
    torch.manual_seed(17)
    vocab = {"[UNK]": 0, "[EOS]": 1, "[BOS]": 2, "hello": 3, "world": 4,
             **{f"t{i}": i for i in range(5, 64)}}
    core = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    core.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=core, unk_token="[UNK]", eos_token="[EOS]", bos_token="[BOS]")
    architectures = [
        ("gpt2", GPT2LMHeadModel, GPT2Config(vocab_size=64, n_positions=64, n_embd=32, n_layer=1, n_head=2,
                                            eos_token_id=1, bos_token_id=2)),
        ("llama", LlamaForCausalLM, LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                                              num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                                              max_position_embeddings=64, eos_token_id=1, bos_token_id=2)),
    ]
    results = []
    for name, model_class, config in architectures:
        model_path = output / "models" / name
        model_class(config).eval().save_pretrained(model_path, safe_serialization=True)
        tokenizer.save_pretrained(model_path)
        for dtype in ("float32", "bfloat16"):
            tag = name + "-" + dtype
            directory = create_workspace(project / "templates/engine/transformers", output / "engines" / tag)
            write_json(directory / "engine.json", {"backend": "transformers", "model_path": str(model_path),
                "device": "cpu", "model_loader": "causal_lm", "max_context": 64, "num_threads": 1,
                **precision_fields(normalize_precision({"compute_dtype": dtype}))})
            engine = EngineProcess(paths, tag, sys.executable, startup_timeout=45, request_timeout=15)
            try:
                client = engine.start(directory, output / tag / "engine.log")
                reference = capture_reference(client, [{"id": "normal", "prompt": "hello world", "max_new_tokens": 4}],
                    output / tag / "reference.json", {"vocab_size": 64, "random_weights": True},
                    {"logit_samples": 64}, {"prompts": ["hello"], "max_new_tokens": 4})
                report = evaluate_correctness(client, reference, {"atol": 0, "rtol": 0, "concurrency": 2})
                assert report["passed"], report
                results.append({"model": name, "precision": dtype, "passed": True, "random_weights": True,
                                "agent_reasoning_tested": False, "coverage": report["coverage"]})
            finally:
                engine.stop()
    write_json(output / "report.json", {"passed": True, "device": "cpu", "results": results})


if __name__ == "__main__":
    main()
