# Robust decode-latency bench via STREAMING: measures inter-token latency (ITL) directly,
# per token, over a long generation -> no prefill-subtraction, no negative/degenerate cells.
# Reports median ms/tok + [p10,p90] spread + per-rep medians (reproducibility) + TTFT.
import requests, time, statistics, random, sys
port = int(sys.argv[1]); label = sys.argv[2]; ctx = int(sys.argv[3])
ngen = int(sys.argv[4]) if len(sys.argv) > 4 else 200
REPS = 3


def run():
    ids = [random.randint(10, 100000) for _ in range(ctx)]
    t0 = time.time()
    r = requests.post(
        f"http://127.0.0.1:{port}/generate",
        json={"input_ids": ids, "sampling_params": {"max_new_tokens": ngen, "temperature": 0},
              "stream": True},
        stream=True, timeout=1800,
    )
    ts = []
    for line in r.iter_lines(decode_unicode=True):
        if line and line.startswith("data:"):
            if line[5:].strip() == "[DONE]":
                break
            ts.append(time.time())
    ttft = (ts[0] - t0) * 1000 if ts else float("nan")
    itl = [(ts[i] - ts[i - 1]) * 1000 for i in range(1, len(ts))]  # per-token decode, ms
    return ttft, itl


run()  # warm
medians, allitl, ttfts = [], [], []
for _ in range(REPS):
    ttft, itl = run()
    ss = itl[8:]  # drop the first few tokens (ramp)
    if len(ss) >= 20:
        medians.append(statistics.median(ss)); allitl += ss; ttfts.append(ttft)
if not medians:
    print(f"{label} ctx={ctx}: FAILED (no tokens)"); sys.exit(0)
med = statistics.median(medians)
q = statistics.quantiles(allitl, n=10)
print(f"{label} ctx={ctx}: decode {med:.2f} ms/tok  [p10 {q[0]:.2f}, p90 {q[8]:.2f}]  "
      f"reps={[round(m, 2) for m in medians]}  n={len(allitl)}  TTFT={statistics.median(ttfts):.0f}ms", flush=True)
