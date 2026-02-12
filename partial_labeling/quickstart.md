# Partial Labeling (10% Budget) + kNN Propagation: Standalone Proxy Score Experiment

> 这份 README 是给 Codex/你自己实现代码用的“任务说明书”。  
> 目标：先写一份**相对独立**的实验代码，验证 “只知道 10% 的离散分数 + embedding kNN 传播” 能否生成质量足够好的 `proxy score`（不要求接入现有 `ScoreCurationPipeline`）。

---

## 0. TL;DR（你最终要做的事）

你要写一个**独立的小实验**（Standalone Proxy Scoring Experiment）：

1. 输入：数据集 + embeddings +（可选）全量 teacher label（用于离线评估）。
2. 只保留 10% 的已知分数（seed labels），其余当作 unknown。
3. 用 embedding kNN 传播，把 unknown 补成全量 `bin_score_proxy`。
4. 输出：
   - 每条样本的 `bin_score_proxy` + `proxy_confidence` + `proxy_source`
   - 指标（dev/teacher 模式下）：accuracy / macro-F1 / confusion matrix 等

核心交付物是一个可复现的实验脚本 + 产物目录（proxy 结果 + metrics）。

---

## 1. Motivation（为什么要做）

### 1.1 成本问题
如果要靠 LLM 为每条样本打分（离散分数，比如 `bin_score`），在 RecSys 规模（百万级+）下成本和时延不可接受。  
但 DS2-lite 类型方法又依赖“每条样本都有分数”才能继续做诊断/纠偏/筛选。

### 1.2 现有 DS2-lite 的价值你不想丢
当前 repo 的 pipeline（诊断 + 修正 + 多样性）已经做了：

- embedding 编码（`docta/core/preprocess.py`）
- 估计噪声转移矩阵 `T`（HOC，`docta/core/hoc.py`）
- SimiFeat 检测疑似错标（`docta/apis/detect.py` + `docta/core/knn.py`）
- 基于置信度的 score curation + diversity score（`score_curation/data_curation.py`）

所以要探索一个“预算标注”的替代方案：只标 10%，然后用传播近似全量分数。

### 1.3 新 idea 的假设
假设：embedding 邻域存在一定“分数局部一致性”。  
如果 embedding 只编码语义而不含质量信号，传播可能失败；这需要通过实验先验证。

---

## 2. Repo 现状（你可以直接复用什么）

### 2.1 你这次实验不需要接入 DS2-lite
本实验只验证：`10% labels + kNN` 能否预测其余 90% 的 labels。  
后续如果效果好，再把 `bin_score_proxy` 作为 `score_key` 接回 DS2-lite 即可（可选）。

### 2.2 数据预处理（已存在）
`score_curation/data_preprocess.py`：
- 连续分数 `label_score -> bin_score`
- 多字段拼接成 `embed_text`

### 2.3 embedding & kNN（已存在，可借用实现思路）
- embedding：`docta/core/preprocess.py`
- 余弦距离：`docta/core/core_utils.py` (`cosDistance_chunked`)
- DS2 用的 kNN 分布统计：`docta/core/knn.py`
- DS2 的 batched kNN（拿邻居 labels + distances）：`docta/core/hoc.py:get_consensus_patterns`

注意：这些模块目前假设每条样本都有合法 label（`0..K-1`）。  
但你这次实验可以独立写，不需要调用这些诊断/curation 逻辑；只需要借用 embedding 或距离计算实现思路。

---

## 3. 你要写的独立实验：Proxy Score Builder

### 3.1 输入/输出数据契约（Data Contract）

**Input** 至少需要二者之一：
- A) `dataset_path`（json/jsonl/parquet）+ `feature_key=embed_text`：脚本内部计算 embeddings
- B) `embedded_pt`（例如 `results/<dataset>/embedded_<dataset>_0.pt`）：直接读已有 embeddings（更快）

可选：`teacher_score_key`（例如 `bin_score`）用于 dev/teacher 模式离线评估。

**Output**（建议写成 json/jsonl + 一个 metrics 文件）：
- `bin_score_proxy`（int）：`0..K-1`，每条样本必须有
- `proxy_confidence`（float）：传播置信度（例如 `max(p)`）
- `proxy_source`（string）：`seed` / `propagated`

建议额外输出一个元信息文件（便于复现）：
- `proxy_meta.json`：包含 `budget`, `seed`, `K`, `knn_k`, `tau`, `method`, `num_llm_calls`, `llm_model`, `prompt_version` 等

### 3.2 必须支持的两种运行模式（非常重要）

1. **Dev/Teacher mode（无 LLM，快速开发）**
   - 适用于像 `raw_data/timeline_label.json` 这种已经有 `bin_score` 的数据
   - 做法：只保留 `budget` 比例的 `teacher_score_key` 作为 seed labels，其余视为 unknown
   - 目的：完全不接 LLM API，也能端到端评估 “传播是否可行”

2. **Real LLM mode（真实调用 LLM）**
   - subset `L` 调 LLM 输出离散分数
   - 必须支持 caching/resume，避免中断重跑烧钱

---

## 4. 传播算法（先做一个能跑的 baseline）

### 4.1 Baseline：kNN 加权投票（推荐先实现）

对每个未标注样本 `u`：

1. 找 `knn_k` 个最近邻（在全体样本上找邻居）
2. 只取其中属于已标注集合 `L` 的邻居
3. 用距离/相似度做权重，算类别分布 `p_u(c)`

可用的权重形式（任选一个，先简单）：
- `w_i = exp(sim(u,i) / tau)`（sim 越大权重越大）
- 或 `w_i = exp(-dist(u,i) / tau)`（dist 越小权重越大）

输出：
- `bin_score_proxy(u) = argmax_c p_u(c)`
- `proxy_confidence(u) = max_c p_u(c)`

Edge cases：
- 若 `NN_k(u) ∩ L` 为空：
  - fallback 到 `prior`（已标注集的类别先验分布）
  - 或增大 k / 多跳（后续增强）

### 4.2 进阶（先写 TODO，后做）
- 图扩散/label propagation（更稳，但工程更复杂）
- 用 FAISS/ANN 做可扩展 kNN（大数据必需）

---

## 5. 评估你要看到什么（先定义验收标准）

在 `dev_teacher` 模式下（有全量 `teacher_score_key` 可对比），至少输出：

- overall accuracy（`bin_score_proxy` vs `teacher_score_key`）
- macro-F1（类别不平衡时更有意义）
- confusion matrix（看错在哪些档位）
- coverage：seed vs propagated 的数量比例
- （可选）按 `proxy_confidence` 分桶的 accuracy（看看置信度是否可用）

这一步的目的很简单：回答 “10% seeds + kNN 传播，能不能恢复足够多的正确 proxy labels？”

---

## 5.1 Quick Start（当前已实现命令）

### A. 用现成 embedding 快速跑（推荐先用这个）

```bash
python partial_labeling/run_proxy_knn_experiment.py \
  --config template.py \
  --dataset_path raw_data/timeline_label.json \
  --embedded_pt results/timeline_label/embedded_timeline_label_0.pt \
  --output_dir runs/proxy_knn_timeline \
  --feature_key embed_text \
  --teacher_score_key bin_score \
  --num_classes 6 \
  --budget 0.10 \
  --seed 3 \
  --knn_k 20 \
  --tau 0.1 \
  --mode dev_teacher \
  --output_score_key bin_score_proxy
```

### B. 不提供 embedding，脚本内直接算 embedding（更慢）

```bash
python partial_labeling/run_proxy_knn_experiment.py \
  --config template.py \
  --dataset_path raw_data/timeline_label.json \
  --output_dir runs/proxy_knn_timeline_reembed \
  --feature_key embed_text \
  --teacher_score_key bin_score \
  --num_classes 6 \
  --budget 0.10 \
  --seed 3 \
  --knn_k 20 \
  --tau 0.1 \
  --mode dev_teacher \
  --output_score_key bin_score_proxy
```

### C. 输出文件位置

- `runs/proxy_knn_timeline/proxy_dataset.json`
- `runs/proxy_knn_timeline/metrics.json`
- `runs/proxy_knn_timeline/proxy_meta.json`

---

## 6. Implementation TODO（给 Codex 直接照着实现）

> 目标：把 TODO 写到足够“可执行”，每项都有明确产物/验收标准。

### P0（必须先做，保证能端到端跑通）

- [x] 新增独立脚本：`partial_labeling/run_proxy_knn_experiment.py`
  - 输入参数（最小集）：
    - `--config template.py`（复用 embedding 配置）
    - `--dataset_path ...`（json/jsonl/parquet，可选：用于算 embeddings）
    - `--embedded_pt ...`（可选：直接读 embeddings）
    - `--output_dir runs/proxy_knn_exp1`
    - `--feature_key embed_text`
    - `--num_classes K`
    - `--budget 0.10`
    - `--seed 3`
    - `--knn_k 50`
    - `--tau 0.1`
    - `--mode dev_teacher|llm`（先只实现 dev_teacher）
    - `--teacher_score_key bin_score`（dev_teacher 用）
    - `--output_score_key bin_score_proxy`（默认值）
  - 验收：
    - `output_dir/` 下有：
      - `proxy_dataset.json`（或 jsonl）：含 `bin_score_proxy`
      - `metrics.json`：含 accuracy/macro-F1/confusion/coverage
      - `proxy_meta.json`：记录超参（budget/seed/k/tau/method 等）

- [x] Dataset loader：复用 HF `datasets.load_dataset`
  - 目标：支持 `.json` / `.jsonl` / `.parquet`
  - 验收：与 `docta/datasets/RecSys_data.py` 行为一致

- [x] Embedding extraction（不依赖 label）
  - 复用 `docta/core/preprocess.py` 的 encoder（同 `template.py` embedding_model）
  - 产物：一个 `embeddings.npy` 或者直接内存 tensor（由你选）
  - 验收：
    - embedding shape = `(N, D)`
    - 与 DS2-lite embedding 模型一致（同 config）
  - 备注：
    - 为了让脚本“独立可跑”，优先支持直接读取 `embedded_*.pt`（如果用户提供的话）

- [x] Budget sampler
  - 先实现：`random`（后续再做 stratified/cluster）
  - 输入：`N`, `budget`, `seed`
  - 输出：`labeled_indices`
  - 验收：可复现，且比例正确

- [x] Propagation baseline（kNN 加权投票）
  - 输入：`embeddings`, `labeled_indices`, `labeled_scores`, `K`
  - 输出：`bin_score_proxy`, `proxy_confidence`, `proxy_source`
  - 验收：
    - 每条样本输出合法 label `0..K-1`
    - `proxy_confidence` 在 `[0,1]`

- [x] Dev/Teacher mode 评估打印
  - 在 `mode=dev_teacher` 下输出：
    - overall accuracy（`bin_score_proxy` vs `teacher_score_key` 全量）
    - coverage（多少样本来自 propagated vs seed）
  - 验收：在 `raw_data/timeline_label.json` 上能跑出数值

- [x] 文档里给出完整命令（你实现后应能一条命令跑通）
  - `partial_labeling/run_proxy_knn_experiment.py` 一条命令产出 proxy + metrics

### P1（真实 LLM 接入，允许逐步完善）

- [ ] LLM scorer interface：`partial_labeling/llm_scorer.py`（或你喜欢的路径）
  - 输入：样本（id + embed_text）
  - 输出：离散分数 `0..K-1`
  - 要求：
    - 强制 JSON 输出（可 parse）
    - 失败重试 + rate limit
    - caching（`cache.jsonl`，按 sample id 去重）
  - 验收：中断后可 resume，且不会重复计费

- [ ] Prompt spec：`partial_labeling/prompts/score_prompt.md`
  - 包括 score rubric（每档分数定义）
  - 验收：prompt 版本号写入 `proxy_meta.json`

- [ ] 抽样策略增强（可选但很有用）
  - stratified by cluster / uncertainty sampling
  - 验收：同预算下传播效果优于 random（dev_teacher 对比）

### P2（扩展性/性能/稳定性）

- [ ] 可扩展 kNN
  - Option A：FAISS（新增依赖，适合大规模）
  - Option B：torch batched matmul + topk（先能跑，后优化）
  - 验收：百万级能在可接受时间内完成（按你的环境定义）

- [ ] Label propagation（图扩散）
  - 验收：在 dev_teacher 下 accuracy 或 calibration 改善

- [ ] 单元测试：`tests/test_propagation.py`
  - toy embeddings + known labels
  - 验收：shape、概率归一、边界 case 通过

---

## 7. 建议的开发路线（强烈推荐按这个顺序）

1. 先用 `raw_data/timeline_label.json` 做 dev_teacher，完全不碰 LLM API
2. 传播能稳定跑后，再接真实 LLM scorer（可选）
3. 最后做性能（FAISS/ANN）和更强传播（graph diffusion）

---

## 8. 常见坑（提前写清楚，避免你实现时踩雷）

- DS2-lite 的 `run_diagnose` 假设 label 都是 `0..K-1`：不要把未标注用 `-1` 塞进去跑诊断
- embedding 只做语义可能不够：如果传播效果差，要把 quality signals 融入 `embed_text`（或额外特征）
- 大规模 kNN 不要 O(N^2)：先写 baseline，随后切 ANN
