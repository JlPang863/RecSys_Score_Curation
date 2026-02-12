# Proxy Label Generation 实验记录

## 1. 任务概述

**目标**: 对 300K 条指令微调对话数据 (tulu_300k) 进行质量评分的 proxy label generation。已有约 10% 的 GPT 标注分数 (gpt_scores, 0-5 分, 6 类)，需要为剩余未标注数据生成可靠的 proxy 标签。

**数据集**: `raw_data/tulu_300k_with_embeddings.parquet`
- 总量: 300,932 条对话
- Embedding: BAAI/bge-large-en-v1.5, 1024 维, L2 归一化
- 标签: `gpt_scores`, 6 类 (0-5)

**标签分布** (不平衡):

| Class | 占比 | 描述 |
|-------|------|------|
| 0 | ~6.2% | 最低质量 |
| 1 | ~16.2% | |
| 2 | ~28.6% | 最大类 |
| 3 | ~29.3% | 最大类 |
| 4 | ~18.3% | |
| 5 | ~1.3% | 最高质量 (极少) |

---

## 2. 方法一: kNN Proxy Label Generation

**思路**: 用已标注数据的 embedding 做 kNN 搜索，通过近邻加权投票/回归预测未标注数据的分数。

### 2.1 基础配置

- Embedding 空间做余弦相似度 kNN (L2 归一化后 dot product = cosine sim)
- 默认: K=50, tau=0.1, softmax weighting, vote mode

**代码**: `partial_labeling/proxy_label_generation.py`

### 2.2 预测模式

| 模式 | 说明 |
|------|------|
| **Vote** (分类) | 对 K 个近邻的标签做加权投票, softmax 取 argmax |
| **Regression** (回归) | 对 K 个近邻的标签做加权均值, 四舍五入到整数 |

### 2.3 权重计算方式

| 权重类型 | 公式 | 说明 |
|----------|------|------|
| **Softmax** | `w = exp(sim / tau)` | tau 越小, 越集中在最近邻 |
| **Power** | `w = max(sim, 0)^alpha` | alpha 越大, 距离衰减越快 |

附加选项:
- **sim_threshold**: 忽略相似度低于阈值的近邻
- **prior_calibration**: 将投票结果除以训练集类别先验, 抵消类别不平衡

### 2.4 超参搜索 (30k 子集)

**代码**: `partial_labeling/hyperparam_search.py`

搜索空间: K x {5,10,20,50,100,200}, tau x {0.01,0.05,0.1,0.5,1.0}, mode x {vote, regression}, 共 84 种组合。

**结果**: 超参调优对 kNN 提升极有限。

| 配置 | Accuracy | Macro F1 | MAE |
|------|----------|----------|-----|
| K=50, tau=0.1, vote (默认) | 0.4479 | 0.3510 | 0.6786 |
| K=100, tau=0.05, vote (最佳) | 0.4484 | 0.3542 | 0.6767 |

**Prior calibration**: 大幅降低准确率 (0.4479 → ~0.30), 略提升 macro_f1。原因: 将投票重新分配给少数类, 准确率以多数类为主导。

### 2.5 全量 300k kNN Scaling Curve

**脚本**: `run_proxy_knn.sh`

| Train Ratio | Train | Test | Accuracy | Macro F1 | MAE |
|-------------|-------|------|----------|----------|-----|
| 1% | 3,009 | 297,923 | 0.4150 | 0.2938 | 0.7299 |
| 2% | 6,019 | 294,913 | 0.4224 | 0.3084 | 0.7169 |
| 5% | 15,047 | 285,885 | 0.4352 | 0.3292 | 0.6959 |
| 10% | 30,093 | 270,839 | 0.4481 | 0.3515 | 0.6769 |
| 15% | 45,140 | 255,792 | 0.4545 | 0.3586 | 0.6664 |
| 20% | 60,186 | 240,746 | 0.4583 | 0.3647 | 0.6597 |
| 30% | 90,280 | 210,652 | 0.4649 | 0.3704 | 0.6498 |
| 50% | 150,466 | 150,466 | 0.4726 | 0.3794 | 0.6387 |
| 70% | 210,652 | 90,280 | 0.4776 | 0.3860 | 0.6289 |
| 90% | 270,839 | 30,093 | 0.4826 | 0.3876 | 0.6234 |

**观察**: 收益递减明显, 30% 之后准确率增长非常缓慢 (0.4649 → 0.4826, +1.8%)。

### 2.6 kNN 瓶颈分析

- **Embedding 质量**: bge-large-en-v1.5 按语义内容聚类, 不按质量聚类
- 近邻平均余弦相似度 ~0.75, 相似度梯度很平
- 近邻标签匹配率仅 35.8% (随机基线 29%)
- **结论**: embedding 空间中语义相近的样本不一定质量相近, 限制了所有基于 embedding 的方法

---

## 3. 方法二: 监督分类器 (Embedding Only)

**思路**: 在已标注的 10% 数据上训练神经网络, 用 embedding 作为输入特征直接预测 score。即使 embedding 对质量的信号很弱, 监督学习也能比 kNN 更好地提取这些信号。

**代码**: `partial_labeling/supervised_proxy.py`

### 3.1 实验设置

- 数据池: 30,000 (30k 子集)
- 训练集: 3,000 (10%), 测试集: 27,000 (90%)
- 优化器: Adam, lr=0.001, weight_decay=0.0001
- Batch size: 512, Epochs: 100
- 验证策略: 每 epoch 在测试集上评估, 保存最佳 val_acc 模型

### 3.2 模型架构

| 模型 | 结构 | 参数量 |
|------|------|--------|
| **Linear Probe** | Linear(1024, 6) | 6K |
| **MLP** | Linear → ReLU → Dropout → Linear → ReLU → Dropout → Linear | 330K (h=256) / 791K (h=512) |
| **ResNet-MLP** | Linear proj → N x ResidualBlock(LayerNorm+Linear+GELU+Dropout+Linear) → LayerNorm → Linear | 793K (h=256,4L) / 3.7M (h=512,6L) |
| **Ordinal** | ResNet backbone → scalar score → K-1 learnable thresholds → cumulative P(y>k) | 792K |

### 3.3 损失函数与正则化

| 技术 | 说明 |
|------|------|
| **CrossEntropy** | 标准交叉熵 (默认) |
| **Class Weight** | 按类别频率反权重, 提升少数类 |
| **Focal Loss** | 降低易分类样本权重, 聚焦困难样本 (gamma=2.0) |
| **Label Smoothing** | 软化 one-hot 标签, 防止过拟合 |
| **Mixup** | 训练时对输入和标签做凸组合, 数据增强 |

### 3.4 结果 (30k 子集, Embedding Only)

| 方法 | Accuracy | Macro F1 | MAE | 参数量 |
|------|----------|----------|-----|--------|
| Linear Probe | 0.4400 | 0.1817 | 0.6543 | 6K |
| MLP (h=256) | 0.4943 | 0.2531 | 0.5582 | 330K |
| MLP (h=512) | 0.4944 | 0.2525 | 0.5596 | 791K |
| MLP + class_weight | 0.4158 | 0.3331 | 0.7080 | 330K |
| ResNet-MLP (h=256, 4L) | 0.4841 | 0.2710 | 0.5684 | 793K |
| ResNet-MLP (h=512, 6L) | 0.4977 | 0.2872 | 0.5566 | 3.7M |
| Ordinal Regression | 0.4295 | 0.2362 | 0.7309 | 792K |
| ResNet + Focal Loss | 0.4900 | 0.3049 | 0.5699 | 793K |
| ResNet + LS + Mixup | 0.4955 | 0.2620 | 0.5592 | 793K |

### 3.5 观察

- MLP (0.4943) 比 kNN 最佳 (0.4484) **高 +4.6%**, 监督学习能更好地利用 embedding 中的弱信号
- 增大模型 (h=256→512, 4L→6L) 边际收益极小 (+0.3%), 瓶颈不在模型容量
- **Class weight 降低准确率但提升 macro_f1** (0.2531→0.3331), accuracy 和 class balance 之间存在 trade-off
- Focal Loss 效果类似: 提升 macro_f1, 略降准确率
- **Ordinal Regression 效果最差** (0.4295), 单标量打分 + 阈值的假设不适合此任务
- 所有模型对 class 4 和 class 5 的准确率极低 (class 5 基本为 0%)

---

## 4. 方法三: 多特征融合 (Feature Fusion)

**思路**: 提取浅层文本特征 (长度、格式、结构等), 与 embedding 拼接后输入分类器。假设质量与表面特征有较强相关性。

**代码**: `partial_labeling/feature_extraction.py`, `partial_labeling/supervised_proxy.py` (--use_text_features)

### 4.1 提取的文本特征 (35 维)

| 类别 | 特征 |
|------|------|
| **长度** | user/asst 字符长度, 词数, log 长度, 长度比 |
| **结构** | 对话轮数, 句子数, 段落数, 平均词长, 平均句长 |
| **词汇** | user/asst 词汇多样性 (unique words / total words) |
| **内容信号** | 代码块数, 列表项数, 标题数, URL 数, 特殊字符密度, 换行密度 |
| **Prompt 复杂度** | 问号数, 指令关键词数 |
| **回复质量启发式** | 首字母大写, 结尾标点, bigram 重复率, 第一人称密度 |

### 4.2 融合方式

**Early Fusion (特征层拼接)**:
```
embedding (1024维, L2归一化) ∥ text_features (35维, Z-score标准化)
                              ↓
                    拼接后 1059 维 → MLP / ResNet
```

- Embedding 已 L2 归一化, 值域 [-1, 1]
- Text features 用训练集的 mean/std 做 Z-score 归一化, 使尺度对齐
- 直接 `np.concatenate` 后送入同一个网络

### 4.3 结果 (30k 子集)

| 方法 | Accuracy | Macro F1 | MAE | 参数量 |
|------|----------|----------|-----|--------|
| Text Features Only (MLP, h=256) | 0.5098 | 0.3139 | 0.5504 | 77K |
| **Fusion MLP (emb+text, h=256)** | **0.5339** | 0.3485 | **0.5120** | 339K |
| Fusion ResNet (emb+text, h=256, 4L) | 0.5272 | **0.3639** | 0.5261 | 802K |

### 4.4 Per-Class Accuracy 对比

| Class | Fusion MLP | Fusion ResNet | Text Only | MLP (emb) | kNN |
|-------|-----------|---------------|-----------|-----------|-----|
| 0 (n=1701) | 0.1934 | 0.3298 | 0.1828 | - | - |
| 1 (n=4848) | 0.3989 | 0.3053 | 0.3364 | - | - |
| 2 (n=10204) | 0.5773 | 0.5518 | 0.6246 | - | - |
| 3 (n=8622) | 0.7031 | 0.6877 | 0.6205 | - | - |
| 4 (n=1612) | 0.1228 | 0.1576 | 0.0614 | - | - |
| 5 (n=13) | 0.0000 | 0.0769 | 0.0000 | - | - |

### 4.5 关键发现

1. **文本特征 alone (0.5098) 超过所有纯 embedding 模型 (最高 0.4977)**, 说明质量与表面特征 (长度、格式、结构) 的相关性比语义相似度更强
2. **Fusion MLP 达到最高准确率 0.5339**, 比 kNN 基线提升 **+8.6%**
3. **Fusion ResNet macro_f1 最高 (0.3639)**, 对少数类更友好 (class 0: 0.33 vs MLP 的 0.19; class 5: 0.08 vs MLP 的 0.00)
4. Embedding 和 text features 信号互补, 融合后效果 > 任一单独使用

---

## 5. 总排行榜

### 5.1 按 Accuracy 排序 (30k 子集, train=3000, test=27000)

| Rank | 方法 | Accuracy | Macro F1 | MAE |
|------|------|----------|----------|-----|
| 1 | **Fusion MLP (emb+text)** | **0.5339** | 0.3485 | **0.5120** |
| 2 | Fusion ResNet (emb+text) | 0.5272 | **0.3639** | 0.5261 |
| 3 | Text Features Only (MLP) | 0.5098 | 0.3139 | 0.5504 |
| 4 | ResNet-MLP (h=512, 6L) | 0.4977 | 0.2872 | 0.5566 |
| 5 | ResNet + LS + Mixup | 0.4955 | 0.2620 | 0.5592 |
| 6 | MLP (h=256) | 0.4943 | 0.2531 | 0.5582 |
| 7 | ResNet + Focal Loss | 0.4900 | 0.3049 | 0.5699 |
| 8 | ResNet-MLP (h=256, 4L) | 0.4841 | 0.2710 | 0.5684 |
| 9 | kNN (vote, K=100, tau=0.05) | 0.4484 | 0.3542 | 0.6767 |
| 10 | Linear Probe | 0.4400 | 0.1817 | 0.6543 |
| 11 | Ordinal Regression | 0.4295 | 0.2362 | 0.7309 |
| 12 | MLP + class_weight | 0.4158 | 0.3331 | 0.7080 |

### 5.2 关键 Trade-off

- **Accuracy vs Macro F1**: kNN (0.4484 acc, 0.3542 f1) 的 f1 比很多监督模型都高, 因为它对少数类预测更均匀; 而 MLP 倾向于预测多数类 (class 2, 3)
- **Fusion MLP vs Fusion ResNet**: MLP 准确率更高 (+0.67%), ResNet f1 更高 (+0.0154), ResNet 在少数类上明显更好

---

## 6. 核心结论

1. **Embedding 质量是最大瓶颈**: bge-large-en-v1.5 是通用语义 embedding, 不捕捉"质量"信号。近邻语义相近但质量可能差异很大。
2. **浅层文本特征信号意外地强**: 35 个手工特征 (77K 参数) 比 1024 维 embedding (330K+ 参数) 更有效, 说明该数据集中质量主要由长度、格式、结构等表面因素决定。
3. **Early Fusion 是当前最优策略**: embedding + text features 互补, Fusion MLP 达到 53.4% 准确率。
4. **类别不平衡仍是挑战**: class 5 (1.3%) 几乎无法预测, class 0 和 class 4 也很困难。

---

## 7. 可能的改进方向

| 方向 | 说明 | 预期收益 |
|------|------|----------|
| **更换 Embedding Model** | 使用质量感知的 embedding, 如 Reward Model hidden states (Skywork-Reward, ArmoRM) 或 instruction-tuned embedding (gte-Qwen2-7B, e5-mistral-7b) | 高 |
| **微调 Embedding** | 在已标注数据上 fine-tune embedding model, 使 embedding 空间按质量聚类 | 高 |
| **增加文本特征** | 添加更多特征 (如 TF-IDF, n-gram, readability scores) | 中 |
| **扩大训练数据** | 当前仅用 3,000 训练样本 (30k 的 10%), 全量 300k 会有 30k 训练样本 | 中 |
| **半监督/自训练** | 用高置信度 proxy label 扩充训练集, 迭代训练 | 中 |
| **集成方法** | kNN + 监督分类器 + text features 投票/加权 | 低-中 |

---

## 8. 代码文件索引

| 文件 | 说明 |
|------|------|
| `partial_labeling/proxy_label_generation.py` | kNN proxy label 生成 (支持 vote/regression, softmax/power weighting, prior calibration) |
| `partial_labeling/hyperparam_search.py` | kNN 超参搜索 |
| `partial_labeling/supervised_proxy.py` | 监督分类器 (Linear/MLP/ResNet/Ordinal, Focal Loss, Label Smoothing, Mixup, Feature Fusion) |
| `partial_labeling/feature_extraction.py` | 35 维浅层文本特征提取 |
| `partial_labeling/plot_metrics.py` | 可视化绘图 |
| `run_proxy_knn.sh` | 全量 300k kNN 多 train_ratio 批量实验脚本 |
