"""Host-owned tokenizer used for context accounting across engine versions."""

from collections.abc import Mapping


def load_control_tokenizer(config, backend):
    model_path = config.get("model_path")
    if model_path:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False)
        return tokenizer, {"owner": "vibeforge-control", "model_path": model_path}
    tokenizer = getattr(backend, "TOKENIZER", None)
    if tokenizer is not None and config.get("backend") == "toy":
        return tokenizer, {"owner": "fixture", "model_path": None}
    return None, {"owner": "unavailable", "model_path": model_path}


def count_tokens(tokenizer, request):
    if "messages" in request:
        arguments = {
            "tokenize": True,
            "add_generation_prompt": request.get("add_generation_prompt", True),
        }
        if request.get("tools"):
            arguments["tools"] = request["tools"]
        token_ids = tokenizer.apply_chat_template(request["messages"], **arguments)
    else:
        text = request.get("text")
        if not isinstance(text, str):
            raise ValueError("tokenize requires text or messages")
        encoded = tokenizer(text, add_special_tokens=request.get("add_special_tokens", False))
        token_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
    # Transformers 5 returns BatchEncoding by default from apply_chat_template.
    # len(BatchEncoding) is the number of fields, not the number of tokens.
    if isinstance(token_ids, Mapping):
        if "input_ids" not in token_ids:
            raise ValueError("Tokenizer result does not contain input_ids")
        token_ids = token_ids["input_ids"]
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return token_ids
