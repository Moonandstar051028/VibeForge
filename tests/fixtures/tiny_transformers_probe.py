"""Offline CPU integration probe with random tiny weights, never an LLM quality benchmark."""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
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
                diagnostic = client.profile(prompt="hello world", max_new_tokens=4)
                assert diagnostic["profiler"] == "torch.profiler" and diagnostic["top_cpu_ops"], diagnostic
                assert not diagnostic["top_device_ops"], diagnostic
                phases = {phase["name"]: phase for phase in diagnostic["phases"]}
                assert phases["vibeforge.prefill"]["calls"] == 1, phases
                assert phases["vibeforge.decode"]["calls"] == 3, phases
                assert diagnostic["normal_request"]["output_tokens"] == 4, diagnostic
                assert diagnostic["diagnostics"]["shape_capture"] == "bounded_module_forward_hooks"
                assert diagnostic["diagnostics"]["module_input_signatures"].get("mlp"), diagnostic
                if name != "gpt2":
                    assert diagnostic["diagnostics"]["module_input_signatures"].get("full_attention"), diagnostic
                assert "backend.generate" in diagnostic["implementation_index"]["symbols"], diagnostic
                # Profiling hooks must be gone before the next ordinary request.
                after = client.generate(prompt="hello", max_new_tokens=4, fixed_length=True)
                assert after["host_observation"]["output_tokens"] == 4
                assert after["host_observation"]["ttft_seconds"] > 0
                assert after["host_observation"]["tpot_seconds"] > 0
                assert after["host_observation"]["client_ttft_seconds"] is None
                assert after["host_observation"]["phases"]["forward_calls"] == 4
                assert after["host_observation"]["token_timing_available"]
                effective = after["host_observation"]["observed_generate_kwargs"][0]
                assert effective["do_sample"] is False and effective["use_cache"] is True
                assert effective["max_new_tokens"] == effective["min_new_tokens"] == 4
                assert client.health()["runtime"]["execution_path"] == "transformers_generate"
                assert after["token_ids"] == reference["benchmark_cases"][0]["token_ids"]
                results.append({"model": name, "precision": dtype, "passed": True, "random_weights": True,
                                "agent_reasoning_tested": False, "coverage": report["coverage"],
                                "profiler": diagnostic["profiler"], "phases": diagnostic["phases"]})
            finally:
                engine.stop()

    # Exercise the real batching path separately from the serialized reference
    # matrix. Two simultaneous HTTP requests must share one model.generate call,
    # preserve their individual outputs, and report the observed batch size.
    model_path = output / "models" / "gpt2"
    tag = "gpt2-float32-microbatch"
    directory = create_workspace(project / "templates/engine/transformers", output / "engines" / tag)
    write_json(directory / "engine.json", {
        "backend": "transformers", "model_path": str(model_path), "device": "cpu",
        "model_loader": "causal_lm", "max_context": 64, "num_threads": 1,
        "request_scheduler": "microbatch", "microbatch_max_size": 2,
        "microbatch_wait_ms": 50,
        **precision_fields(normalize_precision({"compute_dtype": "float32"})),
    })
    engine = EngineProcess(paths, tag, sys.executable, startup_timeout=45, request_timeout=15)
    try:
        client = engine.start(directory, output / tag / "engine.log")
        prompts = ["hello", "hello world"]
        references = [client.generate(prompt=prompt, max_new_tokens=4, fixed_length=True)
                      for prompt in prompts]
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(client.generate, prompt=prompt, max_new_tokens=4, fixed_length=True)
                       for prompt in prompts]
            concurrent = [future.result() for future in futures]
        assert [item["token_ids"] for item in concurrent] == [item["token_ids"] for item in references]
        assert all(item["host_observation"]["batch_size"] == 2 for item in concurrent), concurrent
        assert all(item["host_observation"]["batching_mode"] == "static_microbatch"
                   for item in concurrent), concurrent
        runtime = client.health()["runtime"]
        assert runtime["request_scheduler"] == "microbatch", runtime
        assert runtime["execution_path"] == "transformers_microbatch_generate", runtime
        assert runtime["batching_mode"] == "static_microbatch", runtime
        assert runtime["continuous_batching"] is False, runtime
        microbatch = {
            "passed": True, "model": "gpt2", "precision": "float32",
            "observed_batch_sizes": [item["host_observation"]["batch_size"] for item in concurrent],
            "outputs_match_serial": True, "runtime": runtime,
        }
    finally:
        engine.stop()
    write_json(output / "report.json", {
        "passed": True, "device": "cpu", "results": results, "microbatch": microbatch})


if __name__ == "__main__":
    main()
