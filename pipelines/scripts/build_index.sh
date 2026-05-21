#!/bin/bash
# Build INDEX.md review packet at artifacts/bases/INDEX.md.
# Usage: build_index.sh {success|failed} [failing_phase]

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
BASE="${REPO_ROOT}/artifacts/bases"
LOGS_DIR="${REPO_ROOT}/logs"
INDEX="${BASE}/INDEX.md"
STATUS="${1:-unknown}"
FAIL_PHASE="${2:-}"

mkdir -p "${BASE}"

cat > "${INDEX}" <<EOF
# Stage 1E CCA vs Water-Filling — Review Packet

**Status:** ${STATUS^^}
**Generated:** $(date '+%Y-%m-%d %H:%M:%S')

EOF

if [[ "${STATUS}" == "failed" ]]; then
    cat >> "${INDEX}" <<EOF
## Failure
Pipeline failed at: \`${FAIL_PHASE}\`

See \`${LOGS_DIR}/pipeline.log\` and any \`*.FAILED\` files under \`${LOGS_DIR}/\` for diagnostics.

EOF
fi

cat >> "${INDEX}" <<EOF
## Headline (b_avg = 3, layer-0-excluded)

**See E3 summaries**: \`${BASE}/e3/*_summary.json\` (search for \`top1_prefill[l0excl_mean]\`).

## E1+E2 — Closed-form simulation
- Spectrum diagnostic: \`figures/spectrum_overlay.png\`, \`figures/r95_heatmap.png\`, \`figures/spectrum_per_layer.png\`
- Pareto frontier: \`figures/sim_pareto.png\`
- Per-layer log-ratios: \`figures/sim_per_layer_lines.png\`
- Per-method heatmaps: \`figures/sim_log_ratio_*_b3.png\`
- Metrics: \`metrics_e1_e2.json\`

## E3 — Real quantization (b_avg ∈ {2, 3, 4}, r=64)
EOF

if [[ -d "${BASE}/e3" ]]; then
    for s in $(ls "${BASE}/e3"/*_summary.json 2>/dev/null); do
        echo "- \`$(basename "${s}")\`" >> "${INDEX}"
    done
fi

cat >> "${INDEX}" <<EOF

## E4a — Cross-task generalization (calibrate on one config, eval on three)
EOF
if [[ -d "${BASE}/e4a" ]]; then
    for s in $(ls "${BASE}/e4a"/*_summary.json 2>/dev/null); do
        echo "- \`$(basename "${s}")\`" >> "${INDEX}"
    done
fi

cat >> "${INDEX}" <<EOF

## E4b — Within-task LOO (24 folds, 8 per config)
EOF
if [[ -d "${BASE}/e4b" ]]; then
    for s in $(ls "${BASE}/e4b"/*_summary.json 2>/dev/null); do
        echo "- \`$(basename "${s}")\`" >> "${INDEX}"
    done
fi

cat >> "${INDEX}" <<EOF

## E5 — Decode-phase Q (pulled from E3 outputs with --query-phase both)

Search the e3 \`*_summary.json\` for \`top1_decode[l0excl_mean]\` vs \`top1_prefill[l0excl_mean]\`.
A small gap means decode-phase queries are well-served by prefill-time calibration.

## Logs
- Pipeline: \`${LOGS_DIR}/pipeline.log\`
- Per-run: \`${LOGS_DIR}/<run_name>.log\` (with \`*.summary.json\` on success or \`*.FAILED\` on failure)
- Registry: \`${LOGS_DIR}/_registry.tsv\`

## What to look at first (5 minutes)

1. \`${BASE}/figures/sim_pareto.png\` — closed-form Pareto across methods.
2. \`${BASE}/e3/e3_b3_r64_summary.json\` — headline real-quantization results.
3. The Stage 1E report under \`notes/stage1e_cca_vs_waterfill_report.md\` (if pipeline succeeded).
EOF

echo "Wrote ${INDEX}"
