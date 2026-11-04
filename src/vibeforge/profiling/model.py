"""Validate safetensors indexes and headers without allocating model weights."""

import json
import struct
import hashlib
from pathlib import Path
from vibeforge.storage.atomic import digest, read_json


def inspect_model(directory):
    directory = Path(directory).resolve()
    config = read_json(directory / "config.json")
    index = read_json(directory / "model.safetensors.index.json")
    mapping = index["weight_map"]
    observed, shards = {}, []
    for filename in sorted(set(mapping.values())):
        path = (directory / filename).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError(f"Missing or invalid weight shard: {filename}")
        size = path.stat().st_size
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError(f"Truncated header: {filename}")
            length = struct.unpack("<Q", prefix)[0]
            if length > 64 * 1024 * 1024 or length + 8 > size:
                raise ValueError(f"Invalid safetensors header: {filename}")
            header = json.loads(stream.read(length))
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            start, end = tensor["data_offsets"]
            if not 0 <= start <= end <= size - 8 - length:
                raise ValueError(f"Invalid tensor bounds: {name}")
            observed[name] = {"file": filename, "dtype": tensor["dtype"], "shape": tensor["shape"]}
        shards.append({"file": filename, "bytes": size})
    missing = [name for name, file in mapping.items() if observed.get(name, {}).get("file") != file]
    if missing:
        raise ValueError(f"Index/header mismatch: {missing[:5]}")
    required = ("tokenizer_config.json", "tokenizer.json")
    for filename in required:
        if not (directory / filename).is_file():
            raise ValueError(f"Missing tokenizer file: {filename}")
    text = config.get("text_config", config)
    token_files = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"]
    token_hashes = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                    for name in token_files if (directory / name).is_file()}
    return {
        "path": str(directory), "model_type": config["model_type"],
        "architectures": config.get("architectures", []), "text_config": text,
        "shards": shards, "tensor_count": len(mapping),
        "weight_bytes": index.get("metadata", {}).get("total_size", sum(s["bytes"] for s in shards)),
        "identity_hash": digest({"config": config, "index": index, "shards": shards, "tokenizer": token_hashes}),
        "tokenizer_hashes": token_hashes,
        "validation": "all_indexed_tensor_headers_and_bounds; not a full weight checksum",
    }
