#!/usr/bin/env python3
"""Boot-echo gate: assert the *resolved* server config matches the protocol.

The serve path has burned us with silently-ignored knobs (CTX passed via env
and dropped; the legacy sampling default). This gate reads back what the
server actually took — /get_server_info, not our launch command — and refuses
the cell on any mismatch.

    echo_gate.py --port 30901 --expect '{"context_length": 73728, ...}' \
                 --out resolved_server_info.json
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request

# server_info keys worth recording even when not asserted on.
RECORD_KEYS = [
    "model_path", "context_length", "chunked_prefill_size",
    "tensor_parallel_size", "tp_size", "mem_fraction_static",
    "max_running_requests", "max_total_num_tokens",
    "triton_attention_num_kv_splits", "kv_cache_dtype",
    "kv_cache_quant_group_size", "disable_radix_cache", "moe_runner_backend",
    "cuda_graph_max_bs", "prefill_attention_backend",
    "decode_attention_backend", "sampling_backend", "random_seed",
    "version",
]


def flatten(obj, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                flatten(v, out)
            elif k not in out:
                out[k] = v
    elif isinstance(obj, list):
        for v in obj:
            flatten(v, out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--expect", required=True, help="JSON dict of key: expected")
    ap.add_argument("--out", help="write the recorded server_info subset here")
    args = ap.parse_args()

    with urllib.request.urlopen(
            f"http://127.0.0.1:{args.port}/get_server_info", timeout=30) as r:
        info = json.load(r)
    flat: dict = {}
    flatten(info, flat)

    resolved = {k: flat[k] for k in RECORD_KEYS if k in flat}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(resolved, f, indent=2, default=str)

    expect = json.loads(args.expect)
    bad = []
    for key, want in expect.items():
        if key not in flat:
            bad.append((key, want, "<absent from server_info>"))
            continue
        got = flat[key]
        mismatch = (abs(got - want) > 1e-6 if isinstance(want, float)
                    else got != want)
        if mismatch:
            bad.append((key, want, got))

    if bad:
        print("ECHO GATE FAILED — server resolved config != protocol:")
        for key, want, got in bad:
            print(f"  {key}: expected {want!r}, server has {got!r}")
        return 1
    print(f"echo gate ok — {len(expect)} knobs verified, "
          f"pool={flat.get('max_total_num_tokens', '?')} tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
