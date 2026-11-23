"""Build an explicit broader validation config; never rewrite an existing run."""
import argparse
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(config, tokenizer):
    config = copy.deepcopy(config)
    snippet = ('Review the following inference code and explain correctness risks, scheduling, '
               'KV cache ownership and concrete tests.\n'
               'def decode_step(hidden, cache):\n'
               '    query = attention(hidden, cache)\n'
               '    hidden = moe(query)\n'
               '    return sample(hidden), update_cache(cache, hidden)\n')
    tokens = tokenizer.encode(snippet, add_special_tokens=False)
    profiles = list(config['workload']['profiles'])
    for name, length, output, concurrency in (
            ('decode_2k_256', 2048, 256, 1),
            ('context_8k_256', 8192, 256, 1),
            ('concurrent4_2k_256', 2048, 256, 4),
            ('decode_2k_1024', 2048, 1024, 1)):
        ids = (tokens * (length // len(tokens) + 1))[:length]
        prompt = tokenizer.decode(ids, skip_special_tokens=True)
        count = len(tokenizer.encode(prompt, add_special_tokens=False))
        profiles.append({'id': name, 'prompts': [prompt], 'max_new_tokens': output,
                         'concurrency': concurrency, 'requests_per_prompt': max(2, concurrency), 'weight': 1,
                         'min_input_tokens': int(count * .98), 'max_input_tokens': int(count * 1.02) + 1})
    config['evaluation']['max_metric_regressions'] = {'latency_mean_seconds': .05, 'ttft_seconds_mean': .05, 'tpot_seconds_mean': .05}
    config['workload']['profiles'] = profiles
    config['workload']['description'] = 'Broader diagnostic matrix: 8k input, up to 1024 output, concurrency 4. Still not a production workload claim.'
    config['profiling']['max_matrix_profiles'] = len(profiles)
    config['profiling']['timeout_seconds'] = 1800
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'configs/qwen3-coder-30b-a3b-a800.json'))
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    config = json.loads(Path(args.config).read_text())
    tokenizer = AutoTokenizer.from_pretrained(config['model']['path'], local_files_only=True)
    output = Path(args.output)
    with output.open('x') as stream:
        stream.write(json.dumps(build(config, tokenizer), ensure_ascii=False, indent=2) + '\n')
