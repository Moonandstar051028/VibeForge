"""Validate safetensors indexes and headers without allocating model weights."""

import json
import struct
import hashlib
import math
from pathlib import Path
from vibeforge.storage.atomic import digest, read_json


DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1,
               "F8_E5M2": 1, "I64": 8, "I32": 4, "I16": 2, "I8": 1,
               "U8": 1, "BOOL": 1}


def inspect_model(directory, model_format="safetensors"):
    if model_format == "gguf":
        return inspect_gguf(directory)
    if model_format == "manifest":
        return inspect_manifest(directory)
    directory = Path(directory).resolve()
    config = read_json(directory / "config.json")
    index_path = directory / "model.safetensors.index.json"
    index = read_json(index_path) if index_path.exists() else {}
    mapping = index.get("weight_map")
    filenames = sorted(set(mapping.values())) if mapping else ["model.safetensors"]
    observed, shards = {}, []
    dtype_elements, dtype_bytes = {}, {}
    for filename in filenames:
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
            if name in observed:
                raise ValueError(f"Duplicate tensor name: {name}")
            observed[name] = {"file": filename, "dtype": tensor["dtype"], "shape": tensor["shape"]}
            if any(not isinstance(d, int) or d < 0 for d in tensor["shape"]):
                raise ValueError(f"Invalid tensor shape: {name}")
            dtype = tensor["dtype"]
            if dtype in DTYPE_BYTES and math.prod(tensor["shape"]) * DTYPE_BYTES[dtype] != end - start:
                raise ValueError(f"Tensor shape/dtype byte count mismatch: {name}")
            dtype_elements[dtype] = dtype_elements.get(dtype, 0) + math.prod(tensor["shape"])
            dtype_bytes[dtype] = dtype_bytes.get(dtype, 0) + end - start
        shards.append({"file": filename, "bytes": size, "header_hash": digest(header), "mtime_ns": path.stat().st_mtime_ns})
    mapping = mapping or {name: tensor["file"] for name, tensor in observed.items()}
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
        "format": "safetensors", "dtype_elements": dtype_elements, "dtype_bytes": dtype_bytes,
        "prequantized": bool(config.get("quantization_config")),
        "shards": shards, "tensor_count": len(mapping),
        "weight_bytes": sum(dtype_bytes.values()),
        "identity_hash": digest({"config": config, "index": index, "shards": shards, "tokenizer": token_hashes}),
        "tokenizer_hashes": token_hashes,
        "validation": "all_indexed_tensor_headers_and_bounds; not a full weight checksum",
    }


def inspect_gguf(path):
    path = Path(path).resolve()
    if path.is_dir():
        candidates = list(path.glob("*.gguf"))
        if len(candidates) != 1:
            raise ValueError("GGUF input needs one explicit file or a directory containing exactly one GGUF")
        path = candidates[0]
    with path.open("rb") as stream:
        header = stream.read(24)
    if len(header) != 24 or header[:4] != b"GGUF":
        raise ValueError("Invalid GGUF header")
    version, tensors, metadata = struct.unpack("<IQQ", header[4:])
    if version not in (2, 3) or not tensors:
        raise ValueError("Only nonempty GGUF v2/v3 files are recognized")
    value = {"path": str(path), "format": "gguf", "model_type": "custom-gguf",
             "weight_bytes": path.stat().st_size, "tensor_count": tensors,
             "gguf_version": version, "metadata_count": metadata, "prequantized": True,
             "validation": "GGUF header and file size only; adapter must validate tensor metadata"}
    value["identity_hash"] = digest({**value, "mtime_ns": path.stat().st_mtime_ns})
    return value


def inspect_manifest(path):
    path = Path(path).resolve()
    path = path / "model_manifest.json" if path.is_dir() else path
    value = read_json(path)
    if not isinstance(value, dict) or not isinstance(value.get("artifacts"), list) or not value["artifacts"]:
        raise ValueError("Model manifest needs a nonempty artifacts list")
    artifacts = []
    for item in value["artifacts"]:
        artifact = (path.parent / item["path"]).resolve()
        if not artifact.is_relative_to(path.parent) or not artifact.is_file():
            raise ValueError(f"Missing or unconfined model artifact: {item['path']}")
        record = {"path": item["path"], "bytes": artifact.stat().st_size, "mtime_ns": artifact.stat().st_mtime_ns}
        if item.get("sha256"):
            hasher = hashlib.sha256()
            with artifact.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(chunk)
            if hasher.hexdigest() != item["sha256"]:
                raise ValueError(f"Model artifact checksum mismatch: {item['path']}")
            record["sha256"] = item["sha256"]
        artifacts.append(record)
    result = {**value, "path": str(path), "format": "manifest", "artifacts": artifacts,
              "weight_bytes": value.get("weight_bytes", sum(a["bytes"] for a in artifacts)),
              "validation": "artifact sizes/mtime and explicitly supplied SHA256 values"}
    if not isinstance(result["weight_bytes"], int) or result["weight_bytes"] <= 0:
        raise ValueError("Model weight_bytes must be a positive integer")
    result["identity_hash"] = digest(result)
    return result
