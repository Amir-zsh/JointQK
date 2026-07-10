#!/usr/bin/env python3
"""A1 probe for plan3 Thrust A: which page score selects the pages that
realized future queries actually attend?

64-token pages over the first 90% of each row's keys; realized mass from the
row's own last-10% queries (the phase1 decode proxy). Page 0 (sinks) and the
last page (recent window) are ALWAYS kept and charged against the keep
budget. Scorers (page score, higher = keep):

  omega_max / omega_mean   max / mean over page tokens of m = k.mu_q/sqrt(d)
                           (calibration signal — what the press can use)
  quest_true               Quest-style per-page coord min/max boxes scored
                           with the TRUE decode queries, mean over queries
                           (needs the query at decode time -> A3 press only)
  quest_mu                 the same boxes scored with calibration mu_q
  incontext_mu             omega_mean with the row's OWN mean decode query
                           in place of mu_q (drift diagnostic; uses realized
                           queries' mean only — measurement, not a press)
  oracle                   realized mass itself (upper bound)
  random_page              seeded random ranking (budget-line control)

Metric: recall@f for f in {5, 10, 25, 50}% = fraction of realized attention
mass captured by the kept pages. Selection rows drive the two FROZEN
decision rules (written into the output before any F1):
  score_mode  = argmax mean recall@25% among non-oracle static scorers
  quest gate  = quest_true beats the best static scorer by > 10 recall
                points @25% on layers 8-31 (fires -> A3 press variant runs)
Posthoc rows are measurement-only (OOD transfer check).

Also exports artifacts/page_quant2/omega_stats_llama31_8b.pt (mu_q per
(layer, kv_head)) — the OmegaPagePress statistic.

Hard sanity asserts: recall curves monotone in budget, oracle >= every
scorer, random_page within 6 points of the budget line.

    python -u pipelines/page_quant/probe_page_selection.py --device cuda:0
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import _bootstrap  # noqa: E402,F401

sys.path.insert(0, str(REPO / "entropy_coding"))

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402

from pipelines.ec.fit_ec_bundle import ROLES  # noqa: E402
from pipelines.page_quant.phase1_empirics import (  # noqa: E402
    EC_RAW, POSTHOC_RAW, load_mu_from_fit_stats, raw_path_any,
)
from kvq.io import save_json, ensure_dir  # noqa: E402

OUT_DIR = REPO / "artifacts/page_quant2"
D = 128
PTOK = 64
N_QUERIES = 512
FRACS = [0.05, 0.10, 0.25, 0.50]
GATE_LAYERS = range(8, 32)
GATE_MARGIN = 10.0            # recall points @25%, pre-registered
SEED = 20260708
# DEPLOYABLE = valid OmegaPagePress score_modes (calibration stats only).
# incontext_mu uses the row's own realized queries — measurement-only: it
# informs the claim ladder but can never be frozen as the press mode or
# serve as the quest-gate baseline.
DEPLOYABLE_SCORERS = ["omega_max", "omega_mean", "quest_mu"]
STATIC_SCORERS = DEPLOYABLE_SCORERS + ["incontext_mu"]
ALL_SCORERS = STATIC_SCORERS + ["quest_true", "oracle", "random_page"]


def page_scores(k, q, mu_q, rng):
    """k (T90, d), q (n_q, d) -> dict scorer -> (n_pages,) scores, plus the
    per-page realized mass. T90 is page-aligned."""
    n_pages = k.shape[0] // PTOK
    kp = k.reshape(n_pages, PTOK, D)

    logits = (q @ k.T) / math.sqrt(D)
    mass = torch.softmax(logits, dim=1).mean(0)
    page_mass = mass.reshape(n_pages, PTOK).sum(1)

    m = (k @ mu_q) / math.sqrt(D)
    mp = m.reshape(n_pages, PTOK)
    kmin = kp.min(1).values                       # (n_pages, d)
    kmax = kp.max(1).values

    def quest(qv):                                # qv (n, d) -> (n_pages,)
        s = torch.maximum(qv.unsqueeze(1) * kmin.unsqueeze(0),
                          qv.unsqueeze(1) * kmax.unsqueeze(0)).sum(2)
        return s.mean(0) / math.sqrt(D)

    q_mean = q.mean(0)
    return {
        "omega_max": mp.max(1).values,
        "omega_mean": mp.mean(1),
        "quest_true": quest(q),
        "quest_mu": quest(mu_q.unsqueeze(0)),
        "incontext_mu": (kp.reshape(-1, D) @ q_mean).reshape(
            n_pages, PTOK).mean(1) / math.sqrt(D),
        "oracle": page_mass,
        "random_page": torch.rand(n_pages, generator=rng).to(k.device),
    }, page_mass


def recall_at(scores, page_mass, frac):
    """Contested-mass recall. Pages 0 (sink) and last (recent) are always
    kept, and sinks alone hold 19-84% of raw attention mass — leaving them
    in the metric floors every scorer near ~85%+ and voids the budget-line
    sanity check (observed: random@50% = 89.3 raw). They are therefore
    excluded from BOTH numerator and denominator: measured is the scorer's
    capture of the mass actually in play. n_keep still counts the two
    forced pages, matching the press's budget accounting."""
    n_pages = scores.shape[0]
    n_keep = max(2, math.ceil(frac * n_pages))
    if n_pages <= 2:
        return 1.0
    n_free = min(n_keep - 2, n_pages - 2)
    if n_free <= 0:
        return 0.0
    s = scores.clone()
    s[0] = -torch.inf
    s[n_pages - 1] = -torch.inf
    keep = s.topk(n_free).indices
    contested = (page_mass.sum() - page_mass[0]
                 - page_mass[-1]).clamp_min(1e-12)
    return float(page_mass[keep].sum() / contested)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip-posthoc", action="store_true")
    args = ap.parse_args()
    dev = torch.device(args.device)
    ensure_dir(OUT_DIR)
    rng = torch.Generator().manual_seed(SEED)
    t0 = time.time()

    roles = json.loads(ROLES.read_text())
    mu_q, _ = load_mu_from_fit_stats(roles["fit"])
    torch.save({"mu_q": mu_q, "model_tag": "llama31_8b",
                "source": "load_mu_from_fit_stats(fit18)", "ptok": PTOK},
               OUT_DIR / "omega_stats_llama31_8b.pt")
    print(f"[probe] omega stats -> {OUT_DIR / 'omega_stats_llama31_8b.pt'}")

    rows_sel = [(c, int(i), "train", EC_RAW) for c, i in roles["selection"]]
    rows_post = []
    if not args.skip_posthoc:
        for p in sorted(POSTHOC_RAW.glob("shard_*/*.pt")):
            parts = p.name.split("__")
            rows_post.append((parts[1], int(parts[2][3:]), "test",
                              POSTHOC_RAW))

    # results[group][scorer][frac] -> list over (row, layer, head);
    # gate_rows[scorer] -> recall@25% restricted to layers 8-31
    results = {g: {s: {f: [] for f in FRACS} for s in ALL_SCORERS}
               for g in ("selection", "posthoc")}
    gate_rows = {s: [] for s in ALL_SCORERS}

    for group, rows in (("selection", rows_sel), ("posthoc", rows_post)):
        for config, idx, split, root in rows:
            art = torch.load(raw_path_any(config, idx, split, root),
                             map_location="cpu", mmap=True,
                             weights_only=False)
            T = int(art["prompt_length"])
            t90 = max(PTOK, int(0.9 * T) // PTOK * PTOK)
            gs = art["q_post"].shape[1] // art["k_post"].shape[1]
            L = art["k_post"].shape[0]
            for l in range(1, L):
                for h in range(art["k_post"].shape[1]):
                    k = art["k_post"][l, h, :t90].to(dev).float()
                    q = (art["q_post"][l, h * gs:(h + 1) * gs, t90:T]
                         .to(dev).float().reshape(-1, D))
                    if q.shape[0] > N_QUERIES:
                        sel = torch.randperm(q.shape[0], generator=rng,
                                             device="cpu")[:N_QUERIES].to(dev)
                        q = q[sel]
                    scores, page_mass = page_scores(
                        k, q, mu_q[l, h].to(dev), rng)
                    for s, sc in scores.items():
                        rec = [recall_at(sc, page_mass, f) for f in FRACS]
                        for f, r in zip(FRACS, rec):
                            results[group][s][f].append(r)
                        # monotone-in-budget sanity (forced pages make this
                        # non-strict only)
                        assert all(rec[i] <= rec[i + 1] + 1e-9
                                   for i in range(len(rec) - 1)), \
                            f"recall not monotone: {s} {rec}"
                        if group == "selection" and l in GATE_LAYERS:
                            gate_rows[s].append(rec[FRACS.index(0.25)])
            del art
            print(f"[probe] {group} {config} row{idx} done "
                  f"({time.time() - t0:.0f}s)", flush=True)

    def mean100(vals):
        return 100.0 * sum(vals) / max(len(vals), 1)

    summary = {g: {s: {f"recall@{int(100 * f)}%": mean100(results[g][s][f])
                       for f in FRACS} for s in ALL_SCORERS}
               for g in ("selection", "posthoc")}

    # sanity: oracle dominates, random tracks the budget line
    for g in ("selection", "posthoc"):
        if not results[g]["oracle"][FRACS[0]]:
            continue
        for s in ALL_SCORERS:
            for f in FRACS:
                assert (summary[g]["oracle"][f"recall@{int(100 * f)}%"]
                        >= summary[g][s][f"recall@{int(100 * f)}%"] - 1e-6),\
                    f"oracle beaten by {s} @{f} ({g})"
        rnd50 = summary[g]["random_page"]["recall@50%"]
        assert abs(rnd50 - 50.0) < 6.0, \
            f"random_page @50% far from budget line: {rnd50:.1f} ({g})"

    # frozen decision rules (selection rows only, BEFORE any F1)
    sel25 = {s: summary["selection"][s]["recall@25%"]
             for s in DEPLOYABLE_SCORERS}
    score_mode = max(sel25, key=sel25.get)
    best_static_gate = max(mean100(gate_rows[s]) for s in DEPLOYABLE_SCORERS)
    quest_gate_val = mean100(gate_rows["quest_true"])
    quest_gate_fired = quest_gate_val > best_static_gate + GATE_MARGIN

    out = {
        "meta": {"ptok": PTOK, "n_queries": N_QUERIES, "seed": SEED,
                 "fracs": FRACS, "gate_layers": [8, 31],
                 "gate_margin": GATE_MARGIN,
                 "recall_basis": "contested (sink+recent pages excluded "
                                 "from num+denom; raw basis floored at "
                                 "~89% by sink mass — see recall_at)"},
        "summary": summary,
        "frozen": {
            "score_mode": score_mode,
            "score_mode_rule": "argmax selection recall@25% among "
                               + ",".join(DEPLOYABLE_SCORERS)
                               + " (incontext_mu measurement-only)",
            "quest_gate_fired": bool(quest_gate_fired),
            "quest_true_at25_layers8_31": quest_gate_val,
            "best_static_at25_layers8_31": best_static_gate,
        },
    }
    save_json(OUT_DIR / "page_selection_probe.json", out)
    print(json.dumps(out["summary"]["selection"], indent=2))
    print(f"[frozen] score_mode={score_mode}  quest_gate="
          f"{'FIRED' if quest_gate_fired else 'not fired'} "
          f"({quest_gate_val:.1f} vs static {best_static_gate:.1f})")
    print(f"[done] -> {OUT_DIR / 'page_selection_probe.json'} "
          f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
