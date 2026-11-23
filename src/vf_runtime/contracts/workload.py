"""Frozen workload matrix with backward-compatible single-profile inputs."""

import math
import re


def workload_profiles(workload):
    return workload.get("profiles") or [{"id": "default", "prompts": workload["prompts"],
        "max_new_tokens": workload["max_new_tokens"], "concurrency": workload.get("concurrency", 1),
        "weight": 1, "requests_per_prompt": 1}]


def normalize_workload(workload):
    profiles = workload.get("profiles")
    if profiles is not None and (not isinstance(profiles, list) or not profiles):
        raise ValueError("workload.profiles must be a nonempty list")
    if profiles:
        first = profiles[0]
        for key in ("prompts", "max_new_tokens", "concurrency"):
            workload.setdefault(key, first.get(key, 1 if key == "concurrency" else 16))
    workload.setdefault("max_new_tokens", 16)
    workload.setdefault("concurrency", 1)
    seen = set()
    for profile in workload_profiles(workload):
        name = profile.get("id", "")
        if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", name) or name in seen:
            raise ValueError("Workload profile IDs must be unique safe identifiers")
        seen.add(name)
        prompts = profile.get("prompts")
        if not isinstance(prompts, list) or not prompts or not all(isinstance(p, str) and p.strip() for p in prompts):
            raise ValueError("Each workload profile needs nonempty string prompts")
        for key, default, upper in (("max_new_tokens", 16, 32768), ("concurrency", 1, 64), ("requests_per_prompt", 1, 100)):
            value = profile.setdefault(key, default)
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError(f"Invalid workload profile {key}")
        weight = profile.setdefault("weight", 1)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
            raise ValueError("Profile weights must be positive finite numbers")
        for key in ("min_input_tokens", "max_input_tokens"):
            if key in profile and (type(profile[key]) is not int or profile[key] <= 0):
                raise ValueError("Input token bounds must be positive integers")
        if profile.get("min_input_tokens", 1) > profile.get("max_input_tokens", math.inf):
            raise ValueError("Input token bounds are reversed")
    selected = workload.setdefault("observation_profile", next(iter(seen)) if len(seen) == 1 else workload_profiles(workload)[0]["id"])
    if selected not in seen:
        raise ValueError("Unknown workload.observation_profile")
    return workload


def observation_profile(workload):
    return next(profile for profile in workload_profiles(workload)
                if profile["id"] == workload.get("observation_profile", workload_profiles(workload)[0]["id"]))


def profiling_cases(workload, coverage="selected", limit=8):
    """Select profile/prompt diagnostics in round-robin order under a frozen cap."""
    profiles = workload_profiles(workload)
    if coverage == "selected":
        profiles = [observation_profile(workload)]
    cases = []
    prompt_index = 0
    while len(cases) < limit:
        added = False
        for profile in profiles:
            if prompt_index < len(profile["prompts"]):
                cases.append({"profile": profile, "prompt_index": prompt_index,
                              "prompt": profile["prompts"][prompt_index]})
                added = True
                if len(cases) >= limit:
                    break
        if not added:
            break
        prompt_index += 1
    return cases


def benchmark_cases(workload):
    return [{"id": f"benchmark-{profile['id']}-{i}", "profile_id": profile["id"],
             "prompt": prompt, "max_new_tokens": profile["max_new_tokens"], "fixed_length": True,
             **{key: profile[key] for key in ("min_input_tokens", "max_input_tokens") if key in profile}}
            for profile in workload_profiles(workload) for i, prompt in enumerate(profile["prompts"])]
