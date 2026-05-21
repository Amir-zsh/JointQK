#!/usr/bin/env python3
"""Generate a self-contained HTML report of the Llama JointQK F1-inversion investigation.

Reads numerical results from the JSON artifacts produced over the course of the
investigation and embeds the Σ_Q drift PNGs as base64. Writes a single HTML file
to notes/jointqk_investigation_report.html.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402, F401

import base64
import html
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def b64_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def main():
    # ---- Load data ----
    llama_kmse = json.loads((REPO / "artifacts/calibration/longbench_compact8_qkv_llama31_8b/05_reports/llama_empirical_kmse_top1_top5.json").read_text())
    qwen3_kmse = json.loads((REPO / "artifacts/calibration/longbench_compact8_qkv/05_reports/qwen3_empirical_kmse_top1_top5_b2.json").read_text())
    mech = json.loads((REPO / "artifacts/calibration/mechanism_analysis_bias_attn_kl.json").read_text())
    logit_kl = json.loads((REPO / "artifacts/calibration/logit_kl_llama_k2.json").read_text())
    traj = json.loads((REPO / "artifacts/calibration/decode_trajectory_llama_k2.json").read_text())
    drift = json.loads((REPO / "artifacts/q_distribution_shift/per_task_drift.json").read_text())

    # Pull F1 from disk
    def read_f1_dir(root: Path):
        out = {}
        if not root.exists(): return out
        for d in root.iterdir():
            if not d.is_dir(): continue
            mfs = [m for m in d.rglob("metrics.json") if m.stat().st_size > 0]
            if not mfs: continue
            m = sorted(mfs, key=lambda p: p.stat().st_mtime)[-1]
            try: out[d.name] = float(m.read_text().strip())
            except: pass
        return out

    verify_f1 = read_f1_dir(REPO / "artifacts/bench_llama_verify")
    compact9_f1 = read_f1_dir(REPO / "artifacts/bench_llama_compact9")
    lcconly_f1 = read_f1_dir(REPO / "artifacts/bench_llama_lcconly")

    # Base64 of chart images
    figs_dir = REPO / "notes/figs/q_drift"
    chart_b64 = {p.stem: b64_image(p) for p in figs_dir.glob("*.png")}

    # ---- Build HTML ----
    H = []
    H.append("""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Llama JointQK F1-inversion investigation</title>
<style>
:root { --fg:#222; --muted:#666; --bg:#fafafa; --hl:#0066cc; --warn:#c62828; --good:#2e7d32; }
* { box-sizing: border-box; }
body { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; line-height: 1.55; color: var(--fg);
       max-width: 1100px; margin: 30px auto; padding: 0 24px; background: var(--bg); }
h1 { border-bottom: 3px solid var(--hl); padding-bottom: 8px; margin-top: 18px; }
h2 { margin-top: 38px; color: var(--hl); border-bottom: 1px solid #ccc; padding-bottom: 4px; }
h3 { margin-top: 24px; color: #333; }
.tldr { background: #fff8e1; border-left: 4px solid #f57c00; padding: 12px 18px; margin: 18px 0; border-radius: 4px; }
.verdict { background: #ffebee; border-left: 4px solid var(--warn); padding: 12px 18px; margin: 18px 0; border-radius: 4px; }
.good { color: var(--good); font-weight: 600; }
.bad { color: var(--warn); font-weight: 600; }
.muted { color: var(--muted); }
table { border-collapse: collapse; margin: 14px 0; font-size: 14px; }
th, td { padding: 6px 11px; border: 1px solid #ddd; text-align: right; }
th { background: #eee; font-weight: 600; }
td.tname { text-align: left; font-weight: 500; }
td.ood { color: var(--warn); font-style: italic; }
td.in { color: var(--good); }
.imgcap { font-size: 13px; color: var(--muted); text-align: center; margin-top: 4px; margin-bottom: 18px; }
img.chart { width: 100%; max-width: 1000px; display: block; margin: 18px auto 4px; border: 1px solid #ddd; border-radius: 4px; background: white; }
.code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; background: #eee; padding: 2px 5px; border-radius: 3px; font-size: 13px; }
.numbig { font-size: 24px; font-weight: 700; }
.k-section { background: white; padding: 18px 22px; border-radius: 6px; margin: 16px 0; border: 1px solid #ddd; }
.delta-pos { color: var(--good); }
.delta-neg { color: var(--warn); }
ul.ruled-out li { margin-bottom: 6px; }
.toc a { display: block; padding: 4px 0; }
.metric-box { background: #f0f7ff; border-left: 3px solid var(--hl); padding: 10px 16px; margin: 10px 0;
              border-radius: 4px; font-size: 14px; }
.metric-box h4 { margin: 0 0 6px; color: var(--hl); font-size: 14px; }
.metric-box .formula { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; background: white;
                       padding: 4px 8px; display: inline-block; border-radius: 3px; border: 1px solid #d0e0f0;
                       margin: 4px 0; font-size: 13px; }
.metric-box ul { margin: 6px 0 0; padding-left: 22px; }
.metric-box li { margin-bottom: 3px; }
details { background: #fff; border: 1px solid #ddd; border-radius: 4px; padding: 10px 14px; margin: 12px 0; }
details summary { cursor: pointer; font-weight: 600; color: var(--hl); }
details[open] summary { margin-bottom: 8px; }
</style>
</head><body>

<h1>Llama-3.1-8B JointQK F1-inversion: investigation report</h1>
<p class="muted">Generated 2026-05-14. Investigation logs and source code under
<span class="code">notes/</span> and <span class="code">pipelines/</span>.</p>

<div class="tldr">
<strong>TL;DR.</strong> On Llama-3.1-8B with K=2 V=3 KV-cache compression, JointQK
(calibrated R_sym basis + water-fill bit allocation) wins every K-side fidelity
metric over TurboQuant V3 (random Hadamard + uniform Lloyd-Max) — K-MSE, top-1,
top-5, attention KL, attention output L2, first-decode logit KL — by large
margins. Yet JointQK <strong>loses ~12.5 pp of LongBench F1 on
<span class="code">lcc</span></strong> (code completion) and ~2-3 pp on hotpotqa
and repobench-p. Every per-window basis-mismatch hypothesis we tested
(zero-bit-coord bias, calibration-domain coverage via compact9, Σ_Q top-16
subspace drift) fails to explain the F1 gap. Multi-step decode-trajectory logit-KL
on lcc reveals that JointQK starts more accurate than TurboQuant at decode step
0 (KL 0.073 vs 0.196) but its logit-KL crosses above TurboQuant's by step 5 and
stays above through the 50-token generation window — the F1 gap is a
<strong>multi-step autoregressive error-compounding</strong> phenomenon, not a
single-step compression-quality issue.
</div>

<nav class="k-section toc">
<strong>Sections</strong>
<a href="#setup">0. Setup, methods, and metrics — read this first</a>
<a href="#problem">1. The problem</a>
<a href="#k-fidelity">2. K-fidelity on compact8: JQ wins by huge margins</a>
<a href="#mechanism">3. Mechanism metrics on compact8: JQ wins downstream too</a>
<a href="#logitkl">4. First-decode logit KL on Llama: JQ wins</a>
<a href="#trajectory">5. Decode-trajectory KL: JQ's advantage flips at t≥5</a>
<a href="#compact9">6. Calibration coverage (compact9 + lcc-only): doesn't close the gap</a>
<a href="#qdrift">7. Σ_Q distribution drift: identical across all 8 tasks</a>
<a href="#verdict">8. Verdict and open questions</a>
</nav>

<h2 id="setup">0. Setup, methods, and metrics</h2>

<h3>The compression task</h3>
<p>A transformer LLM's attention layer at inference time builds a <strong>KV cache</strong>:
keys K and values V for every prompt token, used by every future decode-time query.
At long context (10–16K tokens) the KV cache dominates memory. <strong>KV-cache
compression</strong> reduces each (layer, kv-head, token) key/value vector from
fp16 down to a few bits per coordinate, then decompresses on the fly when a future
query arrives.</p>

<p>Our setup: K compressed to <strong>2 bits per coord</strong> (K=2 — the
challenging budget), V compressed to <strong>3 bits per coord</strong> via the same
random-Hadamard scheme on both methods (V=3). Layer 0 is always full-precision
(known anomalous attention sink). We compare two K-compression methods:</p>

<table>
<tr><th>method</th><th>basis</th><th>bit allocation</th><th>data-dependent?</th></tr>
<tr><td class='tname'>JointQK (R_sym)</td>
    <td>eigenvectors of (Σ<sub>Q</sub>Σ<sub>K</sub>+Σ<sub>K</sub>Σ<sub>Q</sub>)/2 fitted on calibration prompts</td>
    <td>water-fill: more bits where Σ<sub>Q</sub>·Σ<sub>K</sub> is larger; some coords get 0 bits</td>
    <td>yes (per-layer, per-kv-head)</td></tr>
<tr><td class='tname'>TurboQuant V3</td>
    <td>random orthogonal (Walsh-Hadamard, fixed seed)</td>
    <td>uniform: every coord gets ⌊b_avg⌋ bits with Lloyd-Max codebook</td>
    <td>no (random per (layer, head) but model-independent)</td></tr>
</table>

<h3>Calibration corpora</h3>
<ul>
<li><strong>compact8</strong>: 400 train prompts pooled across 8 LongBench tasks
(qasper, hotpotqa, musique, qmsum, multi_news, triviaqa, passage_retrieval_en,
repobench-p). Used to fit R_sym on Llama → produces
<span class="code">cca_stats_llama31_8b_longbench_compact8_n400.pt</span>. This is the
"production" basis.</li>
<li><strong>compact9</strong>: 450 prompts = compact8 + 50 lcc (new in this
investigation, built to test whether lcc being OOD explains the F1 inversion).</li>
<li><strong>lcc-only</strong>: 50 lcc train prompts only.</li>
</ul>
<p>All three corpora are evaluated against the same LongBench test split, with the
50 train rows of each task excluded from F1 evaluation.</p>

<h3>Metrics, explained</h3>

<div class="metric-box">
<h4>K-MSE (K reconstruction MSE)</h4>
<p>Mean-squared error between FP K and reconstructed K, per element, averaged
across (layer, kv-head, token, coord).</p>
<div class="formula">K-MSE = mean<sub>(L,h,t,c)</sub> ( K<sub>FP</sub>[L,h,t,c] − K<sub>recon</sub>[L,h,t,c] )²</div>
<p><strong>Lower is better.</strong> Pure reconstruction quality, ignores how K is
actually used in attention.</p>
</div>

<div class="metric-box">
<h4>top-1 retention (attention-score / key-side)</h4>
<p>For every query <code>q</code> in the test prompt, compute attention scores
<code>q·K<sub>FP</sub><sup>T</sup></code> (FP) and <code>q·K<sub>recon</sub><sup>T</sup></code>
(compressed). Top-1 retention = fraction of queries where the argmax key is the
same under FP and compressed.</p>
<div class="formula">top-1 = Pr[ argmax<sub>j</sub> q·K<sub>FP</sub>[j] == argmax<sub>j</sub> q·K<sub>recon</sub>[j] ]</div>
<p><strong>Higher is better.</strong> Measures whether the single most-attended-to key
is preserved. Pre-softmax, pre-V; ignores everything downstream of the attention
scores. Distinct from the <em>next-token top-1</em> metric in §4/§5 (which takes
argmax over the full output vocabulary after running the entire model forward).</p>
</div>

<div class="metric-box">
<h4>top-5 retention</h4>
<div class="formula">top-5 = Pr[ argmax<sub>j</sub> q·K<sub>FP</sub>[j] ∈ top-5 keys under q·K<sub>recon</sub> ]</div>
<p><strong>Higher is better.</strong> More forgiving than top-1; the FP-winning key
just has to remain in the top-5. Closer to what matters for softmax attention
distributions with a few high-mass keys.</p>
</div>

<div class="metric-box">
<h4>K reconstruction bias_frac</h4>
<p><strong>What it answers:</strong> when K compression makes errors, are those
errors <em>shifted in the same direction for every token</em> (a systematic
bias), or are they <em>different directions for different tokens</em> (random
noise that averages out)?</p>

<p><strong>The decomposition.</strong> Pick one (layer, kv-head). Stack the K
reconstruction error for every prompt token into a matrix of shape
(T tokens, d=128 coords). Each row is the error vector for one token. For each
coord, average down the column → get the <strong>bias</strong> vector
<code>bias[c] = mean<sub>t</sub>(err[t, c])</code> of shape (d,). Subtract that
average back out of every row → get the <strong>noise</strong> residual that has
zero mean across tokens by construction.</p>

<div class="formula">err[t, c] = bias[c] + noise[t, c]    where mean<sub>t</sub>(noise[t, c]) = 0</div>

<p>Total squared error energy <strong>‖err‖²<sub>F</sub></strong> splits cleanly into
bias-energy + noise-energy:</p>
<div class="formula">‖err‖²<sub>F</sub> = T·‖bias‖² + ‖noise‖²<sub>F</sub></div>
<div class="formula">bias_frac = T·‖bias‖² / ‖err‖²<sub>F</sub>   ∈ [0, 1]</div>

<p><strong>Reading the number.</strong></p>
<ul>
<li><code>bias_frac ≈ 0</code> → errors look like i.i.d. zero-mean noise across
tokens. Each token's error points in a different direction. <em>Under
softmax(qK)·V, these per-token errors cancel out</em> (large-T averaging).</li>
<li><code>bias_frac ≈ 1</code> → errors are essentially the same direction at
every token. Compression is consistently shifting K toward a particular vector.
<em>Under softmax(qK)·V this looks like an additive constant in score-space</em> —
queries that align with the bias direction systematically over- or under-attend.</li>
</ul>

<p><strong>Concrete toy example.</strong> Say d=2 and we observe 4-token errors:</p>
<div class="formula">err = [[+0.5, +0.3], [+0.5, −0.3], [+0.5, +0.3], [+0.5, −0.3]]</div>
<p>Column means: <code>bias = [+0.5, 0]</code>. Bias-energy =
<code>4·(0.5² + 0²) = 1.0</code>. Total energy =
<code>4·(0.5² + 0.3²) = 1.36</code>. → bias_frac = 1.0/1.36 ≈ <strong>0.74</strong>
(highly systematic in coord 0; pure noise in coord 1).</p>

<p><strong>Where the hypothesis came from / what it killed.</strong> Our pre-experiment
intuition was: <em>JointQK assigns 0 bits to some coords, which means K<sub>recon</sub>
on those coords is always 0 regardless of K<sub>FP</sub>'s value — that should
create a strong systematic bias</em>. TurboQuant rotates K by a fresh Hadamard,
breaking up the per-coord structure, so we expected TQ's bias_frac &lt; JQ's.
Measurement showed the opposite (Llama: <strong>JQ 0.66% vs TQ 8.15%</strong>),
because (i) JQ's water-fill zeros only directions of low Σ<sub>K</sub> variance
(so K is near-zero there anyway → 0-bit error is near-zero), and (ii) TQ's
Lloyd-Max codebook is deterministic, so equally-rotated inputs produce equally-
biased rounding errors. The bias-creates-F1-inversion theory died here.</p>
</div>

<div class="metric-box">
<h4>Attention probability KL</h4>
<p>For each query <code>q</code>, compute the full softmax over keys with FP and
with compressed K. Measure KL divergence (in nats) between the two distributions,
averaged over queries.</p>
<div class="formula">p<sub>full</sub>(j) = softmax(q·K<sub>FP</sub><sup>T</sup>/√d)[j]</div>
<div class="formula">KL = ∑<sub>j</sub> p<sub>full</sub>(j) · log( p<sub>full</sub>(j) / p<sub>recon</sub>(j) )</div>
<p><strong>Lower is better.</strong> Captures the full softmax distribution distortion,
not just the argmax. A query that attends 50%/50% to two keys is more sensitive
than top-1 reveals.</p>
</div>

<div class="metric-box">
<h4>Attention output relative L2 error</h4>
<p>The actual signal the next layer sees is <code>softmax(qK)·V</code>. Compare
that vector under FP-K vs compressed-K (V is the same in both — V compression is
fixed). Report relative L2.</p>
<div class="formula">attn_out_FP = softmax(q·K<sub>FP</sub><sup>T</sup>/√d) · V</div>
<div class="formula">attn_out_recon = softmax(q·K<sub>recon</sub><sup>T</sup>/√d) · V</div>
<div class="formula">rel-L2 = ‖attn_out_FP − attn_out_recon‖² / ‖attn_out_FP‖²</div>
<p><strong>Lower is better.</strong> 0 = identical, 1 = noise-level wrong. This is the
closest pre-projection signal to what the rest of the transformer actually receives.</p>
</div>

<div class="metric-box">
<h4>First-decode logit KL</h4>
<p>Prefill the prompt under K-compression, then take a single decode forward pass
over the question tokens. At the position that produces the first answer token,
compare the full-vocab softmax under FP vs compressed K. Report KL divergence over
the ~128k vocabulary, averaged over prompts.</p>
<div class="formula">KL(FP→M) = KL( softmax(logits<sub>FP</sub>) ‖ softmax(logits<sub>M</sub>) )</div>
<p><strong>Lower is better.</strong> This is the next-token prediction error after
all downstream layers have processed the compressed attention outputs. It's the
highest-resolution single-step signal for "did compression change what the model
would generate".</p>
</div>

<div class="metric-box">
<h4>Next-token top-1 (logit-level)</h4>
<p>Same setup as first-decode logit KL, but reduce the distribution to its argmax
token id and check equality with FP's argmax. Reported as the fraction of prompts
where the compressed model would have greedy-generated the same next token as FP.</p>
<div class="formula">next-tok top-1 = Pr[ argmax<sub>v</sub> logits<sub>FP</sub>[v] == argmax<sub>v</sub> logits<sub>M</sub>[v] ]</div>
<p><strong>Higher is better.</strong> The argmax-only view of the same logit
distribution that the KL metric covers in full. Discards probability-mass info
in exchange for a "would greedy sampling differ?" yes/no per prompt. Useful as a
sanity check that the KL number translates into actual sampling behavior.</p>
<p><em>This is what was called <code>match_JQ</code>/<code>match_TQ</code> in
early drafts and in the trajectory-probe script — same operation, renamed for
consistency with key-side "top-1 retention" in §2. The two top-1 metrics differ
only in what the argmax is taken over: <code>q·K</code> for the K-fidelity
metric, full-vocab logits for this one.</em></p>
</div>

<div class="metric-box">
<h4>Decode-trajectory logit KL</h4>
<p>Same as first-decode logit KL, but extended over many decode steps under
<strong>teacher forcing</strong>: at each step t, all three caches (FP, JQ, TQ) are
fed FP's argmax token from step t−1, so they all see identical conversation
history. We then measure each method's per-step logit-KL to FP at every t.</p>
<p><strong>Lower is better at each t.</strong> Reveals whether per-step errors stay
bounded or grow / cross over each other across decode steps.</p>
</div>

<div class="metric-box">
<h4>Σ<sub>Q</sub> top-16 subspace cosine</h4>
<p>Compute Σ<sub>Q</sub> = E[q·q<sup>T</sup>] per (layer, kv-head) on either a window
of prefill tokens or a bin of decode-step samples. Take its top-16 eigenvectors
(d=128 head dim). Measure principal-angle Frobenius similarity to the top-16
eigenvectors of the production calibration Σ<sub>Q</sub> reference (pooled compact8).</p>
<div class="formula">cosine = ‖V<sub>window</sub><sup>T</sup> V<sub>ref</sub>‖<sub>F</sub> / √16</div>
<p><strong>1.0 = same top-16 subspace as the calibration reference; lower = drifted.</strong>
Top-16 was chosen because R_sym concentrates ~all energy in the first ~16 eigvecs
on Llama (effective rank). Whether or not this subspace is preserved is what
predicts whether the calibrated bit-allocation hits the high-information directions.</p>
</div>

<details>
<summary>What's "Mode A" / "fraction=1.0" / "layer-0 excluded"?</summary>
<p><strong>Mode A</strong> = compress prefill K only; new K added at decode time
stays at fp16 (no streaming compression). This is the production setting on both
methods in phase-7-v7.</p>
<p><strong>fraction=1.0</strong> = evaluate on the full LongBench test split for
each task (~150 prompts/task after excluding calibration train rows). Earlier
sweeps used fraction=0.5 for speed; v7 ran at fraction=1.0.</p>
<p><strong>Layer-0 excluded</strong> = layer 0 of the transformer has anomalous
attention sink behavior (its Σ_K has huge dynamic range). Both methods keep layer
0 at fp16 to avoid disproportionate damage; all aggregations in this report
exclude layer 0.</p>
</details>

""")

    # === 1. PROBLEM ===
    H.append('<h2 id="problem">1. The problem</h2>')
    H.append("""
<p>F1 = LongBench downstream task quality (per-task F1/ROUGE evaluator). Both
compression methods are evaluated on the same prompts at K=2 V=3, with FP
("full precision", no compression) as the upper-bound reference.</p>

<p><strong>"F1 inversion"</strong> = the situation where method A wins every K-side
reconstruction metric over method B (top-1, top-5, K-MSE — all defined in §0)
but loses to method B on actual F1. On the same model, with the same compression
budget, the metric we use to <em>predict</em> compression quality says A &gt; B
but the metric we actually <em>care about</em> says A &lt; B. This is what
happens with JointQK vs TurboQuant on Llama-3.1-8B and is the puzzle this report
investigates.</p>

<p>On <strong>Qwen3-8B</strong> there is no inversion — the K-side metrics rank
JQ &gt; TQ and F1 agrees (+2.66 pp 11-task mean, +6.26 pp on hotpotqa at K=2 V=3).
On <strong>Llama-3.1-8B</strong> the same K-side ranking holds, but F1 inverts:
JQ loses ~2 pp on average and ~12 pp on a few tasks. The local re-verification
confirmed the inversion is not a remote-machine artifact:</p>
""")

    H.append("<table><tr><th>task</th><th>FP</th><th>TQ-K2</th><th>JQ-base-K2</th><th>JQ−TQ</th></tr>")
    tasks_v = ["hotpotqa", "lcc", "repobench-p", "qasper", "qmsum", "2wikimqa"]
    for t in tasks_v:
        fp = verify_f1.get(f"full_precision_{t}")
        tq = verify_f1.get(f"turboquant_k2_v3_{t}")
        jq = verify_f1.get(f"jointqk_k2_v3_{t}")
        if None in (fp, tq, jq): continue
        d = jq - tq
        cls = "delta-pos" if d > 0 else "delta-neg"
        H.append(f"<tr><td class='tname'>{t}</td>"
                 f"<td>{fp:.2f}</td><td>{tq:.2f}</td><td>{jq:.2f}</td>"
                 f"<td class='{cls}'>{d:+.2f}</td></tr>")
    H.append("</table>")

    H.append("""<p>Headline anomaly: <strong>lcc JQ 35.58 vs TQ 48.05 (−12.5 pp)</strong>.
JQ also loses 2–3 pp on hotpotqa/repobench-p. Wins only on qmsum and 2wikimqa.</p>""")

    # === 2. K-FIDELITY ===
    H.append('<h2 id="k-fidelity">2. K-fidelity on compact8 calibration: JointQK wins by huge margins</h2>')
    H.append("""
<p>Per-(layer, kv-head) empirical K reconstruction quality, measured on
<strong>80 Llama test prompts</strong> from the compact8 calibration test split,
layer-0 excluded. JointQK = <em>A</em>; TurboQuant = <em>C</em>. K compressed at 2 bits.</p>
<p class="muted">Recap: <a href="#setup">top-1</a> = fraction of queries where the
argmax key under FP and under compressed-K agree. <a href="#setup">top-5</a> =
fraction where the FP-winning key remains in the top-5 of compressed-K scores.
<a href="#setup">K-MSE</a> = per-element mean-squared error of K. Higher is better
for top-k retention; lower is better for K-MSE.</p>""")

    H.append("<h3>Llama overall (80 test prompts)</h3>")
    H.append("<table><tr><th>method</th><th>top-1 ↑</th><th>top-5 ↑</th><th>K-MSE ↓</th></tr>")
    for m, name in [("A","JointQK"),("C","TurboQuant")]:
        o = llama_kmse["overall"][m]
        H.append(f"<tr><td class='tname'>{name}</td>"
                 f"<td>{o['top1']:.4f}</td><td>{o['top5']:.4f}</td><td>{o['mse']:.4f}</td></tr>")
    H.append("</table>")

    H.append("""<p><span class="good">JointQK dominates every K-side metric on every task</span> — top-1 by ~17 pp,
top-5 by ~10 pp, K-MSE at 47% of TurboQuant's. This is exactly the K-quality story
that historically motivated the calibrated basis.</p>""")

    H.append("<h3>Per-task (Llama compact8 test prompts)</h3>")
    H.append("<table><tr><th>task</th>"
             "<th>JQ top-1</th><th>TQ top-1</th><th>Δ</th>"
             "<th>JQ top-5</th><th>TQ top-5</th><th>Δ</th>"
             "<th>JQ K-MSE</th><th>TQ K-MSE</th></tr>")
    for t, td in llama_kmse["per_task"].items():
        a = td["A"]; c = td["C"]
        H.append(f"<tr><td class='tname'>{t}</td>"
                 f"<td>{a['top1']:.4f}</td><td>{c['top1']:.4f}</td>"
                 f"<td class='delta-pos'>+{a['top1']-c['top1']:.4f}</td>"
                 f"<td>{a['top5']:.4f}</td><td>{c['top5']:.4f}</td>"
                 f"<td class='delta-pos'>+{a['top5']-c['top5']:.4f}</td>"
                 f"<td>{a['mse']:.4f}</td><td>{c['mse']:.4f}</td></tr>")
    H.append("</table>")

    H.append("<h3>Qwen3 comparison overall (same b=2)</h3>")
    H.append("<table><tr><th>method</th><th>top-1 ↑</th><th>top-5 ↑</th><th>K-MSE ↓</th></tr>")
    for m, name in [("A","JointQK"),("C","TurboQuant")]:
        o = qwen3_kmse["overall"][m]
        H.append(f"<tr><td class='tname'>{name}</td>"
                 f"<td>{o['top1']:.4f}</td><td>{o['top5']:.4f}</td><td>{o['mse']:.4f}</td></tr>")
    H.append("</table>")

    H.append("""<p>On Qwen3 the K-fidelity story is the same — JQ wins by similar margins —
and the F1 outcome is consistent with it. <strong>Llama K-fidelity therefore
fails to predict Llama F1.</strong> That's the puzzle.</p>""")

    # === 3. MECHANISM METRICS ===
    H.append('<h2 id="mechanism">3. Mechanism metrics on compact8: JQ still wins downstream</h2>')
    H.append("""
<p>If K-fidelity wins don't translate to F1 wins, maybe something between K
reconstruction and the next-layer hidden state is corrupted. We measured three
mechanism-level metrics on 4 prompts × 2 models at K=2:</p>
<p class="muted">Recap (full definitions in §0):
<a href="#setup">bias_frac</a> = fraction of K-error energy in the systematic
per-coord mean (vs per-token noise); high values mean the same coords get
shifted the same direction every token.
<a href="#setup">Attention-prob KL</a> = KL of softmax-over-keys distribution
between FP-K and compressed-K.
<a href="#setup">Attention-out rel-L2</a> = relative L2 error of
<code>softmax(qK)·V</code> — what the next layer actually receives. Lower is
better on all three.</p>""")

    H.append("<table><tr><th>model</th><th>method</th>"
             "<th>K bias_frac</th><th>attn-prob KL ↓</th><th>attn-out rel-L2 ↓</th></tr>")
    for model, r in mech.items():
        for m, name in [("A","JointQK"),("C","TurboQuant")]:
            o = r["overall"][m]
            H.append(f"<tr><td class='tname'>{model}</td><td>{name}</td>"
                     f"<td>{o['bias_frac']:.4f}</td>"
                     f"<td>{o['kl_mean']:.4f}</td>"
                     f"<td>{o['attn_out_relerr_mean']:.4f}</td></tr>")
    H.append("</table>")

    H.append("""
<p>Two findings, both bad news for the "JQ has worse downstream behavior" theory:</p>
<ol>
<li><strong>JQ's K reconstruction error has LESS systematic bias than TQ's.</strong>
   TurboQuant's deterministic Hadamard rotation + Lloyd-Max produces 16–23×
   more per-coord bias than JointQK's calibrated allocation (Qwen3: 10.8% vs
   0.5%; Llama: 8.2% vs 0.7%). The early hypothesis that JQ's zero-bit coords
   create systematic per-direction bias is <em>opposite to what the data shows</em>.</li>

<li><strong>JQ's attention-output reconstruction is far cleaner than TQ's.</strong>
   On Llama, JointQK's attention-output relative L2 is 0.092 (≈9% of the FP
   attn output); TurboQuant's is 0.991 — essentially noise-level. JQ wins by
   ~11× on Llama and ~6× on Qwen3.</li>
</ol>
<p>So between K reconstruction and <span class="code">softmax(qK)·V</span>, every metric
agrees: JointQK is dramatically better. If the F1 gap is real, it must come from
something downstream of the attention-output L2.</p>
""")

    # === 4. LOGIT KL ===
    H.append('<h2 id="logitkl">4. First-decode logit KL on Llama (K=2 V=3, 20 prompts/task)</h2>')
    H.append("""<p>Closer to F1: at the first decode position, measure
<span class="code">KL(softmax(logits_FP) || softmax(logits_M))</span> over the full
vocab (~128k tokens). Setup: prefill the prompt with each method's K compression,
then run a single decode pass over the question tokens. Take the next-token logit
distribution at the position that produces the first answer token. Compare to
the FP logits at the same position. Lower KL = closer to FP's next-token
prediction.</p>""")

    H.append("<table><tr><th>task</th><th>n</th>"
             "<th>KL(FP→JQ) ↓</th><th>KL(FP→TQ) ↓</th><th>ΔKL (JQ−TQ)</th>"
             "<th>next-tok top-1 JQ</th><th>next-tok top-1 TQ</th></tr>")
    for t, td in logit_kl["tasks"].items():
        a = td["aggregate"]
        d = a["mean_kl_fp_jq"] - a["mean_kl_fp_tq"]
        cls = "delta-pos" if d < 0 else "delta-neg"
        H.append(f"<tr><td class='tname'>{t}</td><td>{a['n']}</td>"
                 f"<td>{a['mean_kl_fp_jq']:.4f}</td>"
                 f"<td>{a['mean_kl_fp_tq']:.4f}</td>"
                 f"<td class='{cls}'>{d:+.4f}</td>"
                 f"<td>{a['top1_match_jq_frac']:.3f}</td>"
                 f"<td>{a['top1_match_tq_frac']:.3f}</td></tr>")
    H.append("</table>")

    H.append("""<p>On <strong>lcc</strong> — the worst F1 disconnect — JointQK's first-decode
logit-KL is <strong>0.074 vs TurboQuant's 0.178</strong> (2.4× tighter to FP).
JointQK matches FP's top-1 token on <strong>100% of lcc prompts</strong> vs
TurboQuant's 85%. <span class="bad">Yet on the same prompts, JQ's F1 is 12.5 pp
below TQ's.</span> The discrepancy between "first-token correctness" and "5-50
token answer correctness" is what we have to explain.</p>""")

    # === 5. TRAJECTORY ===
    H.append('<h2 id="trajectory">5. Decode-trajectory logit KL: the smoking gun</h2>')
    H.append("""<p>Same setup as §4, extended to many decode steps under
<strong>teacher forcing</strong>: at each step t, all three caches (FP, JQ, TQ)
are fed FP's argmax token from step t−1, so they all see identical conversation
history. We then measure each method's per-step logit-KL to FP at every t, plus
the per-step <a href="#setup">next-token top-1</a> (fraction of prompts where the
method's argmax over the full vocab matches FP's). This isolates "did K
compression change the next-token distribution at this step?" from the question
of which token each method itself would generate. 15 prompts/task, 50 decode
steps. The table below shows lcc — the worst F1 disconnect — sampled every 5
steps:</p>""")

    lcc_traj = traj["tasks"]["lcc"]["per_step_agg"]
    H.append("<table><tr><th>t (decode step)</th>"
             "<th>KL(FP→JQ)</th><th>KL(FP→TQ)</th><th>ΔKL (JQ−TQ)</th>"
             "<th>next-tok top-1 JQ</th><th>next-tok top-1 TQ</th></tr>")
    show_ts = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    for s in lcc_traj:
        if s["t"] not in show_ts: continue
        d = s["mean_kl_jq"] - s["mean_kl_tq"]
        cls = "delta-pos" if d < 0 else "delta-neg"
        H.append(f"<tr><td>{s['t']}</td>"
                 f"<td>{s['mean_kl_jq']:.4f}</td>"
                 f"<td>{s['mean_kl_tq']:.4f}</td>"
                 f"<td class='{cls}'>{d:+.4f}</td>"
                 f"<td>{s['frac_match_jq']:.2f}</td>"
                 f"<td>{s['frac_match_tq']:.2f}</td></tr>")
    H.append("</table>")

    H.append("""<p>At step 0 (the first decode token), JQ leads TQ by 0.123 nats. By step 5,
JQ is <em>behind</em> TQ. By step 10–25, JQ's KL is 2–4× larger than TQ's.
<strong>The F1 inversion lives in autoregressive compounding.</strong> Per-step
JQ errors don't cancel — they accumulate. TQ's random-Hadamard noise apparently
self-cancels under softmax+V across decode positions; JQ's calibrated-error
structure does not.</p>""")

    # === 6. COMPACT9 ===
    H.append('<h2 id="compact9">6. Calibration coverage (compact9 + lcc-only basis)</h2>')
    H.append("""<p><strong>The hypothesis.</strong> The production calibration (compact8) was
fit on 8 LongBench tasks; <code>lcc</code> was not one of them. lcc has the
shortest contexts on average (~3K tokens), empty <code>question</code> field, and
purely-code content. Maybe its query/key distribution is genuinely different from
the QA/summary tasks in compact8, in which case R_sym's bit allocation is
mis-targeted and reconstruction quality on lcc's specific Q/K patterns is worse
than the on-task K-fidelity numbers suggest.</p>
<p><strong>The test.</strong> Capture Llama Q/K/V on lcc prompts (50 train, 10
test), then build two new R_sym bases:</p>
<ul>
<li><strong>compact9</strong>: pooled basis fit on compact8 ∪ {lcc} = 450 train prompts</li>
<li><strong>lcc-only</strong>: basis fit on just 50 lcc train prompts (worst case for
in-domain QA tasks, best case for lcc)</li>
</ul>
<p>Re-ran JQ K=2 V=3 on 4 representative tasks. FP and TQ baselines re-used
from the local verify run. <strong>Prediction if calibration coverage explains the
gap:</strong> JQ-compact9 closes &gt;5 pp of the −12.5 pp lcc gap; JQ-lcconly
closes even more (specialist basis on lcc); other tasks regress by &lt;1 pp.</p>""")

    H.append("<table><tr><th>task</th>"
             "<th>FP</th><th>TQ</th>"
             "<th>JQ-base</th><th>JQ-compact9</th><th>JQ-lcconly</th>"
             "<th>Δ(c9 − base)</th><th>Δ(lcc − base)</th>"
             "<th>JQ-c9 vs TQ</th></tr>")
    tasks_c9 = ["lcc", "repobench-p", "hotpotqa", "2wikimqa"]
    for t in tasks_c9:
        fp  = verify_f1.get(f"full_precision_{t}")
        tq  = verify_f1.get(f"turboquant_k2_v3_{t}")
        jqb = verify_f1.get(f"jointqk_k2_v3_{t}")
        jqc = compact9_f1.get(f"jointqk_k2_v3_{t}")
        jql = lcconly_f1.get(f"jointqk_k2_v3_{t}")
        if None in (fp, tq, jqb, jqc, jql): continue
        d_c9 = jqc - jqb
        d_lcc = jql - jqb
        d_c9_tq = jqc - tq
        cls_c9 = "delta-pos" if d_c9 > 0 else "delta-neg"
        cls_lcc = "delta-pos" if d_lcc > 0 else "delta-neg"
        cls_tq = "delta-pos" if d_c9_tq > 0 else "delta-neg"
        H.append(f"<tr><td class='tname'>{t}</td>"
                 f"<td>{fp:.2f}</td><td>{tq:.2f}</td>"
                 f"<td>{jqb:.2f}</td><td>{jqc:.2f}</td><td>{jql:.2f}</td>"
                 f"<td class='{cls_c9}'>{d_c9:+.2f}</td>"
                 f"<td class='{cls_lcc}'>{d_lcc:+.2f}</td>"
                 f"<td class='{cls_tq}'>{d_c9_tq:+.2f}</td></tr>")
    H.append("</table>")

    H.append("""<div class="verdict"><strong>Calibration-coverage hypothesis falsified.</strong>
Adding lcc to calibration (compact9) closes only <strong>+0.4 pp</strong> on lcc.
Even an <em>lcc-only</em> basis closes only +2.3 pp of the 12.5 pp gap to TQ.
Both bases <em>regress</em> hotpotqa and 2wikimqa by 2–11 pp. The disconnect is
not primarily a calibration-domain coverage issue.</div>""")

    # === 7. Σ_Q DRIFT ===
    H.append('<h2 id="qdrift">7. Σ_Q distribution drift: identical across all 8 tasks</h2>')
    H.append("""<p>Last shot at a basis-mismatch story. For each of 8 tasks (6 in-calib + 2 OOD),
compute window Σ<sub>Q</sub> = E[q·q<sup>T</sup>] from prefill (200-token windows × 5 test
prompts) and from decode (20 prompts × up to 100 decode steps via
<code>model.generate</code> with a RoPE hook). Compute
<a href="#setup">top-16 subspace cosine</a> between window Σ<sub>Q</sub> and the
production compact8 reference, averaged across (L≥1, h).</p>

<p><strong>What this metric tells us in plain terms.</strong> R_sym's bit allocation
puts more bits where Σ<sub>Q</sub>·Σ<sub>K</sub> is largest — i.e., into the top
eigenvectors of Σ<sub>Q</sub>. If a task's Σ<sub>Q</sub> top eigenvectors don't
match the calibration reference, then bits are concentrated in the "wrong"
directions and reconstruction quality on that task's queries should suffer.
Top-16 captures most of the energy of Σ<sub>Q</sub> on Llama (head-dim 128, but
effective rank ≪ 128). Subspace cosine 1.0 means same top-16 directions as
calibration; 0 means orthogonal. Whether lcc has lower cosine than in-calibration
tasks would directly support a basis-mismatch explanation.</p>""")

    H.append(f'<img class="chart" src="data:image/png;base64,{chart_b64["summary_bars"]}" alt="summary bars">')
    H.append('<div class="imgcap">Summary: per-task mean Σ<sub>Q</sub> drift (prefill vs decode) vs compact8 reference.</div>')

    H.append('<img class="chart" src="data:image/png;base64,{}" alt="prefill drift">'.format(chart_b64["prefill_drift"]))
    H.append('<div class="imgcap">Prefill drift across 200-token windows. Each panel = one task; line = mean across 5 test prompts.</div>')

    H.append('<img class="chart" src="data:image/png;base64,{}" alt="decode drift">'.format(chart_b64["decode_drift"]))
    H.append('<div class="imgcap">Decode drift binned by step range. n = total decode-Q samples in the bin.</div>')

    H.append('<img class="chart" src="data:image/png;base64,{}" alt="combined trajectory">'.format(chart_b64["combined_trajectory"]))
    H.append('<div class="imgcap">Combined prefill→decode trajectory. Solid lines = prefill (200-token windows); dotted squares = decode-step bins.</div>')

    H.append("""<p>What the charts show:</p>
<ol>
<li><strong>Prefill cosine clusters at 0.63–0.65 across ALL 8 tasks.</strong> lcc
   (0.63) is statistically indistinguishable from in-calibration repobench-p
   (0.63) or musique (0.64). 2wikimqa (OOD, but JQ <em>wins</em> by +4 pp on F1)
   scores 0.64 — same as the rest.</li>
<li><strong>Within-prefill drift is essentially zero.</strong> Cosine is flat
   across positions 0 → 16K tokens on every task.</li>
<li><strong>Decode cosine is consistently HIGHER than prefill cosine</strong>
   (0.66–0.76 vs 0.63–0.65). Opposite direction of what "decode-Q drifts away
   from prefill-Q" would predict.</li>
</ol>

<p><strong>Caveat on the absolute level.</strong> All tasks scoring ~0.64 (well
below 1.0) likely reflects a metric-noise floor: 200 tokens × 4 q-heads = 800
samples per (L, h) is thin for a d=128 second-moment estimate, so top-16 eigvecs
are partially noise-driven. Decode bins beat prefill because they pool more
samples (~4000 per (L, h)) and yield less noisy Σ_Q estimates. The relative
across-task ranking is what matters — and lcc is not an outlier.</p>

<div class="verdict"><strong>Σ_Q drift hypothesis falsified.</strong> Top-16
subspace alignment of prefill or decode Σ_Q to the calibration reference does
not separate F1-winning tasks from F1-losing tasks. lcc and 2wikimqa — both OOD,
opposite F1 outcomes — score identically.</div>""")

    # === 8. VERDICT ===
    H.append("<h2 id=\"verdict\">8. Verdict — what we ruled out, what's left</h2>")
    H.append("""<p>Summarizing the chain of falsified hypotheses:</p>

<table><tr><th>hypothesis</th><th>verdict</th><th>evidence</th></tr>
<tr><td class='tname'>K reconstruction worse for JQ on Llama</td>
    <td class='good'>FALSE</td><td>JQ wins K-MSE/top-1/top-5 by huge margins (§2)</td></tr>
<tr><td class='tname'>JQ K-error has more systematic bias than TQ</td>
    <td class='good'>FALSE — opposite is true</td><td>JQ bias_frac 0.7% vs TQ 8.2% (§3)</td></tr>
<tr><td class='tname'>JQ attention output corrupts the next-layer input</td>
    <td class='good'>FALSE</td><td>JQ attn-out rel-L2 0.092 vs TQ 0.99 (§3)</td></tr>
<tr><td class='tname'>JQ first-decode logit is more off than TQ's</td>
    <td class='good'>FALSE</td><td>JQ logit-KL 0.074 vs TQ 0.178 on lcc (§4)</td></tr>
<tr><td class='tname'>Zero-bit coords cause structural per-direction drift</td>
    <td class='bad'>PARTIAL — fix HURTS</td><td>min_bits=1 closes only +3.85 pp on lcc, regresses other tasks by 14–16 pp</td></tr>
<tr><td class='tname'>Calibration-domain mismatch (lcc OOD)</td>
    <td class='good'>FALSE</td><td>compact9 closes only +0.4 pp; lcc-only basis only +2.3 pp (§6)</td></tr>
<tr><td class='tname'>Within-prefill Σ_Q drift</td>
    <td class='good'>FALSE</td><td>Cosine flat across all positions (§7)</td></tr>
<tr><td class='tname'>Within-decode Σ_Q drift</td>
    <td class='good'>FALSE — opposite</td><td>Decode-bin cosine HIGHER than prefill (§7)</td></tr>
<tr><td class='tname'>Top-16 Σ_Q subspace mismatch separates tasks</td>
    <td class='good'>FALSE</td><td>All 8 tasks score 0.63–0.65; lcc not an outlier (§7)</td></tr>
</table>

<h3>What's actually true</h3>
<p>The decode-trajectory KL data (§5) is the only signal that points the right
direction. JQ wins at step 0 and loses by step 5 on lcc. This means the
mechanism is in <strong>autoregressive error correlation</strong>, not
single-step magnitude or basis alignment. Two candidate explanations consistent
with the evidence:</p>

<ol>
<li><strong>Direction-correlated reconstruction error</strong>. JQ's basis
   concentrates K reconstruction error into a small number of consistent
   directions (the low-bit and zero-bit coords); the same directions get the
   same bias for every prompt token. As decode-time queries shift over
   generation steps to encode the partial answer, projection onto those
   error-concentrated directions creates correlated per-step biases that
   accumulate constructively. TQ's random Hadamard spreads K error uniformly
   across all 128 directions, so per-step errors are roughly independent and
   average out under softmax+V.</li>

<li><strong>W<sub>O</sub> interaction</strong>. The attention output L2 measure
   (§3) is in the pre-projection basis. The model only sees W<sub>O</sub> @
   attn_out. If W<sub>O</sub> happens to suppress the directions where TQ's
   noise lives and pass directions where JQ's structured error lives, the
   residual-stream error could rank JQ &gt; TQ even though raw attn-out rank is
   JQ ≪ TQ. We have not measured the W<sub>O</sub>-projected error.</li>
</ol>

<h3>Recommended next experiments</h3>
<ol>
<li>Compute <strong>cross-position autocorrelation of K reconstruction error</strong>
   for JQ vs TQ. If JQ has high positive autocorrelation and TQ doesn't, that
   directly confirms the "correlated noise compounds" mechanism.</li>
<li>Measure <strong>W<sub>O</sub>-projected attn-output error</strong>:
   ‖W<sub>O</sub> (softmax(qK_FP)V − softmax(qK_recon)V)‖.</li>
<li>Test a <strong>per-step Hadamard randomization</strong> hybrid: keep R_sym
   for the bit-allocation step but apply a fresh random rotation per
   reconstruction call. Predicted to recover the random-Hadamard
   error-averaging property while preserving the calibration-quality allocation.</li>
</ol>

<p class='muted'>Generated by <span class='code'>analysis/build_jq_investigation_html.py</span>.</p>
</body></html>""")

    out_path = REPO / "notes/jointqk_investigation_report.html"
    out_path.write_text("\n".join(H))
    size_kb = out_path.stat().st_size / 1024
    print(f"wrote {out_path}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
