# Prefill Q/K Stats Collection

- Model: `Qwen/Qwen3-8B`
- Dataset: `longbench-e`
- Configs requested: `qasper,hotpotqa,passage_retrieval_en`
- Examples collected: `75`
- Examples by config: `{'hotpotqa_e': 25, 'passage_retrieval_en_e': 25, 'qasper_e': 25}`
- Length filter: `[4000, 12000]` tokens
- Prompt length stats: min=`4080`, max=`11905`, mean=`7512.0`
- Stored tensors: `q_post`, `k_post` only, prefill positions only
