"""Run with the configured inference interpreter; never load model weights."""

import argparse
import json
import os
from pathlib import Path
from vibeforge.storage.atomic import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--confined-cache", help="Apply the same filesystem write limits as engine workers")
    args = parser.parse_args()
    result = {"model_path": args.model, "model_weights_loaded": False}
    try:
        if args.confined_cache:
            from vibeforge.engine.sandbox import restrict_writes, prepare_device
            cache = Path(args.confined_cache).resolve()
            scratch = cache / 'engine-scratch'
            scratch.mkdir(parents=True, exist_ok=True)
            for key, child in {'TMPDIR': 'engine-scratch', 'CUDA_CACHE_PATH': 'cuda',
                               'TRITON_CACHE_DIR': 'triton', 'HF_HOME': 'huggingface',
                               'TORCH_HOME': 'torch', 'XDG_CACHE_HOME': 'xdg'}.items():
                os.environ[key] = str(cache / child)
            result['device_setup'] = prepare_device('cuda')
            result['confinement'] = restrict_writes([cache])
        import torch
        import accelerate
        from transformers import AutoTokenizer, AutoConfig, AutoModelForImageTextToText
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": "Return JSON with result 4."}],
                                               tokenize=False, add_generation_prompt=True, enable_thinking=False)
        config = AutoConfig.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
        with torch.device("meta"):
            model = AutoModelForImageTextToText.from_config(config, trust_remote_code=False)
        result.update(torch=torch.__version__, accelerate=accelerate.__version__,
                      model_class=type(model).__name__,
                      meta_parameter_count=sum(parameter.numel() for parameter in model.parameters()),
                      cuda_available=torch.cuda.is_available(), device=torch.cuda.get_device_name(0),
                      cuda_sum=torch.ones(8, device="cuda").sum().item(),
                      template_tokens=len(tokenizer.encode(prompt)), passed=True)
    except Exception as error:
        result.update(passed=False, error=f"{type(error).__name__}: {error}")
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
