# Qwen3.5 eager CUDA Graph engine

Specialized for the locally validated Qwen3.5-27B / Transformers 5.20.2 / A100 / NF4-BF16 environment. The recurrent single-token helper replays unchanged eager operations. Prefill remains eager; graph buffers are isolated through copies, a lock, and stream events. Unsupported shapes fall back.

The source verifies that the original helper does not read kwargs before ignoring forwarded metadata. Other Transformers versions require renewed correctness and performance validation. Never replace this with torch.compile without repeating exact-token gates.

This is a Codex-assisted implementation. See docs/10-continuous-improvement.md and the isolated validation report for evidence and scope. Continue optimization with configs/campaigns/qwen35-27b-optimized-codex.json.
