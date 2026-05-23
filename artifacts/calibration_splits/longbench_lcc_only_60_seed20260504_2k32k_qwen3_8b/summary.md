# LongBench Calibration Split

- Dataset: `longbench`
- Model/tokenizer: `Qwen/Qwen3-8B`
- Seed: `20260504`
- Token filter: `[2000, 32000]`
- Tasks: `lcc`
- Split: `50` train + `10` test per task

Training row IDs are calibration examples and should be removed from end-to-end evaluation.

| Task | Selected | Min tokens | Mean tokens | Max tokens | Train row IDs | Test row IDs |
|---|---:|---:|---:|---:|---|---|
| `lcc` | 60 | 2004 | 3973.1 | 22835 | `[139, 10, 160, 381, 310, 104, 63, 494, 408, 352, 289, 211, 184, 189, 26, 164, 443, 334, 163, 259, 390, 492, 477, 286, 254, 489, 179, 109, 322, 151, 491, 222, 423, 150, 172, 439, 340, 221, 362, 248, 359, 412, 320, 261, 414, 410, 297, 467, 11, 103]` | `[438, 490, 95, 30, 384, 419, 302, 43, 4, 64]` |
