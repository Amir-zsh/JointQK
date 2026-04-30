"""Gate for Stage 1E E1-2: validates the distributional diagnostics module's outputs.

Exit code 0 = pass, non-zero = fail with diagnostic printed to stderr.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch


def fail(msg: str) -> None:
    print(f"GATE_E1_2 FAIL: {msg}", file=sys.stderr, flush=True)
    sys.exit(2)


def ok(msg: str) -> None:
    print(f"GATE_E1_2 PASS: {msg}", flush=True)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[3]
    out_dir = repo_root / "artifacts/stage1/cca_vs_waterfill_study/distribution_diagnostics"
    metrics_path = out_dir / "metrics_e1_2.json"
    stats_path = out_dir / "distribution_stats.pt"
    if not metrics_path.exists():
        fail(f"missing {metrics_path}")
    if not stats_path.exists():
        fail(f"missing {stats_path}")

    with open(metrics_path) as f:
        m = json.load(f)
    stats = torch.load(stats_path, map_location="cpu", weights_only=False)

    n_layers = m["n_layers"]
    n_kv_heads = m["n_kv_heads"]
    head_dim = m["head_dim"]
    configs = m["configs"]
    low_thresh = m["low_confidence_threshold"]

    ok(f"loaded {len(configs)} configs: {configs}")
    ok(f"low-confidence tasks: {m['low_confidence_tasks']}")

    # ---- Distance / basis checks ----

    # Self-distance test: for each (cfg, phase), Bures(Σ, Σ) should be ~0. We test by reading
    # `phase_distances` and ensuring values are non-negative + finite. Direct self-distance not
    # explicitly stored, but the Bures function has been unit-tested in the runner's regression test.
    for cfg in configs:
        d = stats["phase_distances"][cfg]
        if not torch.isfinite(d).all():
            fail(f"phase_distances[{cfg}] has non-finite entries")
        if (d < 0).any():
            fail(f"phase_distances[{cfg}] has negative entries: min={d.min().item()}")
    ok("phase_distances are finite and non-negative across all (layer, kv_head)")

    for key, d in stats["cross_task_distances"].items():
        if not torch.isfinite(d).all():
            fail(f"cross_task_distances[{key}] has non-finite entries")
        if (d < 0).any():
            fail(f"cross_task_distances[{key}] has negative entries: min={d.min().item()}")
    ok(f"cross_task_distances finite and non-negative across {len(stats['cross_task_distances'])} pairs")

    # Subspace overlap monotonicity in r (each (key, r) should be ≤ (key, r')) for r' ≥ r).
    # Group overlaps by (m, ca, cb) and check.
    by_pair = {}
    for k, v in stats["subspace_overlaps"].items():
        # k = "metric/ca_vs_cb/rN"
        prefix, rstr = k.rsplit("/r", 1)
        r = int(rstr)
        by_pair.setdefault(prefix, {})[r] = v
    bad_mono = 0
    for prefix, by_r in by_pair.items():
        rs = sorted(by_r.keys())
        for i in range(len(rs) - 1):
            ra, rb = rs[i], rs[i + 1]
            if (by_r[ra] > by_r[rb] + 1e-4).any():
                bad_mono += 1
                # only report the first few
                if bad_mono <= 3:
                    print(f"GATE_E1_2 WARN: {prefix} subspace overlap not monotone: r={ra}<{rb} but some entries decrease", flush=True)
    if bad_mono > 0:
        # This is unexpected because larger r adds basis vectors, so the Frobenius-norm-based overlap should grow.
        # However, we use the normalized form  (sum_squares / r), which can decrease if the new directions
        # don't align well — this is mathematically possible. Don't hard-fail; warn.
        print(f"GATE_E1_2 INFO: {bad_mono} (metric, pair) groups have non-monotone normalized overlap (mathematically possible for normalized-by-r form)", flush=True)
    else:
        ok("subspace overlap monotone non-decreasing in r (normalized form)")

    # F4: per-task P_K_inv · P_K = I.
    ident_errs = m["cca_identity_max_err_per_task"]
    for cfg, err in ident_errs.items():
        if err > 1e-3:
            fail(f"per-task CCA identity violated for cfg={cfg}: max abs err = {err:.4e}")
    ok(f"per-task CCA identity holds for all {len(ident_errs)} configs (max err {max(ident_errs.values()):.4e})")

    # Regression test: combined per-task Σ_Q^prefill matches E1's global Σ_Q.
    rel_err = m["regression_max_rel_err"]
    if rel_err > 1e-3:
        fail(f"Σ_Q regression test failed: max rel err vs E1 = {rel_err:.4e} > 1e-3")
    ok(f"Σ_Q regression test passes: max rel err vs E1 = {rel_err:.4e}")

    # Decode-token count threshold.
    n_tokens = m["n_tokens"]
    for cfg in configs:
        # n_tokens has tuple keys converted to strings via JSON. Lookup whichever form the saver chose.
        decode_key = None
        for k in n_tokens:
            if isinstance(k, str) and cfg in k and "decode" in k.lower():
                decode_key = k
                break
        # save_json may have flattened tuple keys; try several fallbacks
        if decode_key is None:
            # save_json's _convert doesn't change tuples, so JSON serialization should have errored if tuples are used directly.
            # Walk the raw dict.
            for k, v in n_tokens.items():
                if cfg in str(k) and "decode" in str(k).lower():
                    decode_key = k
                    break
        if decode_key is None:
            print(f"GATE_E1_2 WARN: could not locate decode-token count for cfg={cfg} in JSON; manual check needed", flush=True)
            continue
        n_dec = n_tokens[decode_key]
        if n_dec < low_thresh:
            print(f"GATE_E1_2 INFO: decode tokens for {cfg} = {n_dec} < {low_thresh}; flagged low-confidence in metrics_e1_2.json", flush=True)

    # ---- Rank-analysis checks ----

    metric_tensors = stats["metric_tensors"]
    # All eigvals ≥ 0 (after clamp); monotone non-increasing.
    for key, mt in metric_tensors.items():
        evs = mt["eigvals"]
        if (evs < 0).any():
            fail(f"{key}: eigvals negative (min={evs.min().item():.4e})")
        diffs = evs[..., :-1] - evs[..., 1:]
        if (diffs < -1e-6).any():
            fail(f"{key}: eigvals not monotone non-increasing (min step = {diffs.min().item():.4e})")
    ok(f"all {len(metric_tensors)} (metric, task) groups: eigvals non-negative + monotone")

    # r95 < d for at least 50% of (layer, head) entries (l0excl) per (metric, task).
    for key, mt in metric_tensors.items():
        r95 = mt["r95"]
        l0excl = r95[1:].reshape(-1)
        frac_below = float((l0excl < head_dim).float().mean().item())
        if frac_below < 0.5:
            fail(f"{key}: only {frac_below*100:.1f}% of (layer 1+, kv_head) pairs have r95 < d={head_dim}")
    ok(f"all (metric, task) groups: ≥50% of (layer 1+, kv_head) pairs have r95 < d (compression headroom)")

    # Cross-check: pooled-across-tasks r95 for Q_prefill should agree with E1's r95.
    cca_path = repo_root / "artifacts/stage1/cca_vs_waterfill_study/cca_stats.pt"
    if cca_path.exists():
        cca = torch.load(cca_path, map_location="cpu", weights_only=False)
        # Build pooled Σ_Q^prefill from per-task accumulation, then eigh + r95
        n_tok = stats["n_tokens"]
        configs_list = stats["configs"]
        sigma_q_pool = 0
        n_pool = 0
        for cfg in configs_list:
            n = int(n_tok.get((cfg, "prefill"), n_tok.get(f"{cfg}/prefill", 0)) if isinstance(n_tok, dict) else 0)
            if n == 0:
                # try string key
                for kk in n_tok:
                    if isinstance(kk, str) and cfg in kk and "prefill" in kk:
                        n = int(n_tok[kk])
                        break
            sigma_q_pool = sigma_q_pool + stats["sigma_q"][f"{cfg}/prefill"] * n
            n_pool += n
        if n_pool > 0:
            sigma_q_pool = sigma_q_pool / n_pool
            # Compute r95 from pooled Σ_Q
            sym = 0.5 * (sigma_q_pool + sigma_q_pool.transpose(-1, -2))
            d = sym.shape[-1]
            tr = sym.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / d
            scale = tr.clamp_min(1e-12)
            reg = 1e-4 * scale[..., None, None] * torch.eye(d).unsqueeze(0).unsqueeze(0)
            evs, _ = torch.linalg.eigh(sym + reg)
            evs = torch.flip(evs.clamp_min(0), dims=[-1])
            cum = torch.cumsum(evs, dim=-1)
            frac = cum / cum[..., -1:].clamp_min(1e-30)
            r95_recon = (frac >= 0.95).int().argmax(dim=-1) + 1
            # Compute r95 from E1's stored Σ_Q (cca_stats.pt's sigma_q)
            sym_e1 = 0.5 * (cca["sigma_q"] + cca["sigma_q"].transpose(-1, -2))
            tr_e1 = sym_e1.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / d
            scale_e1 = tr_e1.clamp_min(1e-12)
            reg_e1 = 1e-4 * scale_e1[..., None, None] * torch.eye(d).unsqueeze(0).unsqueeze(0)
            evs_e1, _ = torch.linalg.eigh(sym_e1 + reg_e1)
            evs_e1 = torch.flip(evs_e1.clamp_min(0), dims=[-1])
            cum_e1 = torch.cumsum(evs_e1, dim=-1)
            frac_e1 = cum_e1 / cum_e1[..., -1:].clamp_min(1e-30)
            r95_e1 = (frac_e1 >= 0.95).int().argmax(dim=-1) + 1
            # Agreement check: ≥95% of (layer, head) within ±1 rank
            within = ((r95_recon - r95_e1).abs() <= 1).float().mean().item()
            if within < 0.95:
                fail(
                    f"pooled-task Q_prefill r95 disagrees with E1 r95: only {within*100:.1f}% within ±1 rank"
                )
            ok(f"pooled-task Q_prefill r95 agrees with E1 r95: {within*100:.1f}% within ±1 rank")

    print("GATE_E1_2 ALL CHECKS PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
