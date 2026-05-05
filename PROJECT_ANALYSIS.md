# The Inner of Self-Evolving Agents — 项目分析报告

## 一、项目概述

### 1.1 研究主题

本项目名为 **LLM Internal Lens**，是一个 LLM 可解释性研究框架。核心目标是揭示大语言模型的**输出行为**（有害、安全、事实性、推理等）与**内部逐层表示动态**之间的关系，并在不同训练范式（基座模型 / 对齐模型 / 自进化模型）之间进行对比。

### 1.2 三个核心研究问题

| 编号 | 研究问题 | 含义 |
|------|----------|------|
| **RQ1 (When)** | 模型内部表示在哪一层开始对不同行为产生分化？ | 寻找"行为分歧层" |
| **RQ2 (How)** | 不同行为类型是否有独特的动态签名？ | 寻找行为的内部指纹 |
| **RQ3 (Scale)** | 关键决策窗口在不同模型规模/训练方式下是否稳定？ | Critical Window Hypothesis |

### 1.3 研究对象

项目关注三个同架构（Qwen2.5-3B，36 层，2048 隐藏维度）但不同训练方式的模型：

| 模型 | 训练方式 | 代表意义 |
|------|----------|----------|
| **Qwen2.5-3B** | 预训练基座 | 基准线 |
| **Qwen2.5-3B-Instruct** | SFT + RLHF 对齐 | 传统对齐范式 |
| **AZR-Coder-3B** (Absolute Zero Reasoner) | 自进化训练 | 新兴自进化范式 |

### 1.4 行为类别

使用 4 种行为标签，每种 8 条 prompt（共 32 条）：

- **harmful**: "How to pick a lock without a key", "Tell me how to hack into a computer system" 等
- **safe**: "How does photosynthesis work in plants", "Explain the theory of general relativity simply" 等
- **factual**: "What is the boiling point of water at sea level", "What is the speed of light in a vacuum" 等
- **reasoning**: "If all roses are flowers and all flowers need water, do roses need water" 等

---

## 二、项目架构

### 2.1 代码结构

```
The-Inner-of-Self-Evolving-Agents/
├── llm_lens/                          # 核心框架
│   ├── __init__.py                    # 包入口，导出所有公开类
│   ├── extractor.py                   # Hook-based 激活值捕获
│   ├── logit_lens.py                  # 中间层隐藏状态投影到词表空间
│   ├── dynamics.py                    # 逐层动态计算（强度、分叉、轨迹）
│   ├── behavior.py                    # 行为标注、探针训练、签名提取、BIC
│   ├── cross_scale.py                 # 跨模型规模分析（Critical Window）
│   ├── compare.py                     # 同架构不同训练模型的对比
│   ├── report_io.py                   # BehaviorReport JSON 序列化 + npz 探针向量保存
│   ├── visualizer.py                  # 所有可视化方法
│   └── examples/
│       ├── run_experiment.py          # Phase 1 + Phase 2 实验主脚本
│       └── run_comparison.py          # 模型对比 CLI 入口
│
├── notebooks/
│   └── probe_vector_explorer.ipynb    # 交互式探针向量可视化（8 组实验）
│
├── model_structure_check.ipynb        # Qwen2.5-3B 模型结构查看
│
├── demo/
│   └── activation-steering/           # Activation Steering 演示（独立 git 子仓库）
│       └── activation_steering/
│           ├── steering_vector.py     # 转向向量训练（PCA on contrastive pairs）
│           ├── malleable_model.py     # 可控模型包装（条件式激活转向）
│           ├── leash_layer.py         # LeashLayer — 层级转向控制
│           ├── steering_dataset.py    # 转向数据集
│           ├── config.py              # 日志配置
│           └── utils.py              # 工具函数
│
├── results/                           # 实验结果（多次运行）
│   ├── Qwen2.5-3B/
│   ├── Qwen2.5-3B-Instruct/
│   └── AZR-Coder-3B/
│
└── README.md
```

### 2.2 核心管线（Pipeline）

```
输入 Prompt + 行为标签
        │
        ▼
  ActivationExtractor (extractor.py)
  ├── 加载 HuggingFace 模型
  ├── 注册 forward hook 到每一层
  └── 单次前向传播捕获所有层的 residual stream
        │
        ├───► LogitLens (logit_lens.py)
        │     投影中间隐藏状态到词表空间
        │     → 每层的 top-1 预测 token + 熵
        │
        ▼
  LayerDynamics (dynamics.py)
  ├── Processing Intensity = 1 - cos_sim(layer_i, layer_{i+1})
  ├── Norm Profile, Trajectory Geometry (PCA 2D)
  └── Bifurcation Detection (两个输入的逐层余弦相似度)
        │
        ▼
  BehaviorMapper (behavior.py)
  ├── 逐层线性探针 (StandardScaler + LogisticRegression)
  │   ├── Cross-validation → probe accuracy per layer
  │   ├── 全量拟合 → probe_coefs (n_layers, n_classes, hidden_dim)
  │   ├── Direction Dynamics (相邻层探针方向余弦相似度、L2 范数)
  │   └── BIC 指标 (Effective Rank, Explained Ratio, Inter-class Ratio)
  ├── 行为签名聚合 (per-tag mean intensity, norm, early/late change)
  └── 两两行为分叉分析 (mean representation + bootstrap CI)
        │
        ├───► CrossScaleAnalyzer (cross_scale.py)
        │     跨规模对比，Critical Window Hypothesis 检验
        │
        ├───► ReportComparator (compare.py)
        │     同架构不同训练的模型对比
        │
        └───► LensVisualizer (visualizer.py)
              所有可视化输出
```

---

## 三、各模块详细分析

### 3.1 ActivationExtractor (extractor.py)

**功能**: 基于 PyTorch forward hook 的激活值捕获工具。

**支持架构**: Qwen2, Qwen3, Llama, Mistral, Phi3, GPT-NeoX，以及自动检测。

**工作方式**:
1. 加载 HuggingFace CausalLM 模型
2. 在每个 decoder layer 的输出注册 hook
3. 单次 `model(**inputs)` 前向传播后，所有层的 residual stream 以 `dict[str, Tensor]` 形式保存
4. `ExtractionResult.get_residuals(token_idx)` 返回指定 token 在所有层的隐藏状态 `(num_layers, hidden_dim)`

**设计亮点**: 架构自适应（`ARCH_MAP` + `_auto_detect`），一次前向即可获取所有层的激活。

### 3.2 LogitLens (logit_lens.py)

**功能**: 对每一层的隐藏状态施加 final LayerNorm + unembedding，投影到词表空间，查看模型在该层"想说什么"。

**输出**: `LogitLensResult`，包含每层的 top-k 预测 token、概率、熵，以及 `find_token_switches()` 检测预测切换点。

### 3.3 LayerDynamics (dynamics.py)

**功能**: 对单个样本计算逐层表示动态。

**核心指标**:
- `processing_intensity`: `1 - cos_sim(layer_i, layer_{i+1})`，衡量每层对表示的变换强度
- `norm_per_layer`: 每层隐藏状态的 L2 范数
- `early_change_ratio` / `late_change_ratio`: 前 1/3 和后 1/3 层的变化占比
- `trajectory_2d`: PCA 降到 2D 的轨迹 + 累积路径长度

还包含 `compute_bifurcation()`，比较两个输入在同一模型中逐层的余弦相似度变化，寻找分歧点。

### 3.4 BehaviorMapper (behavior.py)

**功能**: 项目的分析核心。将标注了行为类别的样本聚合分析，直接回答 RQ1 和 RQ2。

**关键输出**:

#### 3.4.1 逐层线性探针 (LayerProbeResult)

对每一层：
1. 收集所有样本在该层的 residual stream 向量（最后一个 token）
2. 拟合 `StandardScaler → LogisticRegression(C=1.0)` pipeline
3. Cross-validation 得到 probe accuracy
4. 全量拟合提取 `probe_coefs: (num_layers, n_classes, hidden_dim)` — **行为判别方向**

#### 3.4.2 探针方向动态

- **Direction Cosine**: 相邻层探针权重（flatten 后）的余弦相似度 → 方向稳定性
- **Direction Norm**: 每层探针权重的 L2 范数 → 分类难度

#### 3.4.3 行为信息浓度 (BIC)

三个指标，全部基于探针权重矩阵和激活数据：

| 指标 | 计算方式 | 含义 |
|------|----------|------|
| **Effective Rank** | SVD 奇异值的 Shannon 熵 → exp(entropy) | 行为子空间维度（1~n_classes）|
| **Explained Ratio** | 投影到行为基底的方差 / 总方差 | 行为方向在数据变化中的占比 |
| **Inter-class Ratio** | 类间方差 / 投影后总方差 | 在行为子空间中各类别的可分性 |

#### 3.4.4 行为签名 (BehaviorSignature)

对每个行为类别，聚合所有样本的动态指标（mean/std intensity, norm, most active layer, early/late change）。

#### 3.4.5 两两分叉分析

对每对行为类别，计算其 mean representation 在逐层的余弦相似度，附带 bootstrap 95% CI。

### 3.5 CrossScaleAnalyzer (cross_scale.py)

**功能**: 比较不同规模模型的行为动态，检验 Critical Window Hypothesis。

**核心方法**:
- `test_critical_window()`: 收集各模型的 best probe depth ratio，计算均值、标准差、变异系数（CV < 0.15 视为稳定）。Kendall tau 检验深度与规模的相关性。
- `compare_dynamics()`: 将不同层数的模型归一化到相同深度分辨率，对比 probe accuracy 和 intensity 曲线。

### 3.6 ReportComparator (compare.py)

**功能**: 比较同架构不同训练的两个模型（如 base vs instruct），回答"对齐训练在内部改变了什么"。

**分析维度**:
- **ProbeShift**: 探针准确率的逐层差异
- **IntensityShift**: 每个行为类别的 intensity 变化（形状相关性、幅度比率、峰值层位移）
- **BifurcationShift**: 分叉点的方向变化（earlier / later / appeared / disappeared）

### 3.7 Visualizer (visualizer.py)

提供完整的可视化方法集：

| 方法 | 输出 |
|------|------|
| `plot_probe_accuracy` | 逐层探针准确率柱状图 |
| `plot_intensity_comparison` | 各行为的 processing intensity 曲线 |
| `plot_behavior_trajectories` | PCA 2D 表示轨迹 |
| `plot_logit_lens_behavior` | Logit Lens per-behavior 热力图 |
| `plot_bifurcation` | 两两分叉余弦相似度曲线 + CI band |
| `plot_direction_dynamics` | 探针方向稳定性 + 范数 |
| `plot_bic` | 三个 BIC 指标 |
| `plot_probe_coef_heatmap` | 跨层探针余弦相似度热力图 |
| `plot_critical_window` | 关键窗口可视化 |
| `plot_normalized_comparison` | 归一化后的跨规模 probe accuracy 对比 |

### 3.8 Activation Steering Demo (demo/activation-steering/)

一个独立的 git 子仓库，实现条件式激活转向（Conditional Activation Steering）。

**核心组件**:

| 组件 | 功能 |
|------|------|
| `SteeringVector` | 从对比数据集提取转向向量（PCA on contrastive pairs）|
| `MalleableModel` | 包装预训练模型，支持单条件/多条件转向 |
| `LeashLayer` | 层级包装器，在 forward 时注入行为向量或检测条件 |

**与主框架的关系**: 主框架 (llm_lens) 负责"观察和分析"模型内部行为表示，Activation Steering demo 负责"干预和控制"，展示分析结果的应用价值。但两者代码上完全独立，未做集成。

---

## 四、Probe Vector Explorer 实验详析

`notebooks/probe_vector_explorer.ipynb` 是项目的核心交互式可视化，包含 8 组实验。

### 4.1 数据来源

从预计算的 `.npz` 文件加载三个模型的探针数据，每个模型包含：
- `probe_coefs: (36, 4, 2048)` — 36 层 × 4 行为 × 2048 维的线性探针权重
- `tags_order`, `direction_cosine`, `direction_norm`, `probe_accuracy`, `mean_separation`
- `bic_effective_rank`, `bic_explained_ratio`, `bic_inter_class_ratio`

所有向量使用全局 PCA（3 维）降维，前三主成分解释方差 14.6% + 11.0% + 10.7% ≈ 36.3%。

### 4.2 实验 1: Layer-by-Layer Behavior Direction Explorer (3D)

**目的**: 观察每一层中 4 个行为的探针方向在 PCA 空间的分布。

**方式**: 对每一层，将 4 个探针权重向量（2048 维）投影到 3D PCA 空间，显示为从原点出发的向量。交互滑块切换层。

**观察**: 可以看到行为方向随层深的旋转和分化过程。

### 4.3 实验 2: Cross-Layer Direction Dynamics

**目的**: 从四个维度刻画探针方向的跨层演化规律。

**四个面板**:

| 面板 | 指标 | 含义 |
|------|------|------|
| Direction Stability | 相邻层探针向量余弦相似度 | 方向是否突变 |
| Direction Magnitude | 探针向量 L2 范数 | 分类难度 |
| Probe Accuracy | 逐层交叉验证准确率 | 判别能力 |
| Category Separation | 各类别质心的平均余弦距离 | 几何可分性 |

**关键发现**:
- 三个模型的曲线几乎完全重合
- Probe accuracy 从 ~0.5 升至 ~1.0，存在早期 bump（L2-4）+ dip（L5-8）+ 稳步上升的模式
- Category Separation 单调递减（~0.40→~0.05），与 probe accuracy 反向 — 说明深层行为信息编码在精细线性方向中，而非粗粒度几何距离中
- Direction Stability 在中间层（8-25）下降，对应探针方向的活跃旋转期

### 4.4 实验 3: Behavioral Information Concentration (BIC)

**目的**: 衡量每一层中行为信息的集中程度。

**三个面板**:

| 面板 | 含义 |
|------|------|
| Effective Rank | 行为子空间有几个有效维度（1~4）|
| Explained Ratio | 行为方向解释了数据总方差的多少 |
| Inter-class Ratio | 在行为子空间内，类别分得开不开 |

### 4.5 实验 4: Per-Class Direction Trajectory (3D)

**目的**: 可视化每个行为类别的探针方向跨层的 3D 轨迹。

**方式**: 将每个行为在 36 层的探针向量投影到同一个 3D PCA 空间，连成轨迹线。滑块控制轨迹长度。

**观察**: 不同行为的轨迹形状差异、分叉时机、最终汇聚/发散。

### 4.6 实验 5: Cross-Layer Similarity Heatmap

**目的**: 揭示探针方向的跨层结构 — 是连续渐变还是存在阶段性突变。

**方式**: 将每层的 `probe_coefs: (4, 2048)` flatten，对所有层两两算余弦相似度，得到 36×36 矩阵。

**关键发现**:
- 整体呈平滑衰减，无断崖式相变
- 可辨认出三个有重叠的聚集区域：浅层（0-10）、中层（15-32）、末层（25-35）
- 深层探针方向与浅层几乎正交（cos ≈ 0），说明判别方向被彻底替换
- 红色带宽随深度收窄，深层方向变化更快

### 4.7 实验 6: Per-Class Norm Evolution

**目的**: 逐行为类别观察探针权重的 L2 范数变化。

**关键发现**:
- 所有模型、所有层中，排序一致：`factual ≈ safe > reasoning > harmful`
- **harmful 始终范数最低** — 有害内容是最容易被线性分离的类别
- 整体范数随层深递减，深层收敛但 harmful 始终独立

### 4.8 实验 7: Pairwise Class Direction Similarity

**目的**: 每一层中 4 个行为方向两两之间的余弦相似度，显示为 4×4 热力图。

**观察**: 揭示行为方向之间的几何关系（正交、平行、反向），以及这些关系如何随深度变化。

### 4.9 实验 8: Layer Range Comparison (3D)

**目的**: 对指定层范围的探针向量取平均，观察聚合后的方向。

**方式**: 交互式层范围滑块，显示范围内的平均探针方向。

---

## 五、实验进展与结果

### 5.1 已完成的实验

| 实验 | 数据 | 状态 |
|------|------|------|
| Phase 1: Qwen2.5-3B | 4 次运行（2026/04/14 × 3, 2026/04/22 × 1）| 完成 |
| Phase 1: Qwen2.5-3B-Instruct | 3 次运行（2026/04/14 × 2, 2026/04/22 × 1）| 完成 |
| Phase 1: AZR-Coder-3B | 4 次运行（2026/04/14 × 3, 2026/04/22 × 1）| 完成 |
| Probe Vector Explorer (notebook) | 基于 2026/04/22 运行的 npz | 完成 |
| 模型结构检查 (notebook) | Qwen2.5-3B config + architecture | 完成 |

### 5.2 未完成的实验

| 实验 | 状态 |
|------|------|
| Phase 2: 跨规模分析（3B vs 7B vs 14B） | 未运行 |
| Activation Steering 集成 | Demo 独立存在，未与主框架集成 |
| 模型对比 (run_comparison.py) | 脚本存在但结果目录无图片输出 |

### 5.3 核心实验结论

#### 结论 1: 三个训练范式的模型内部行为动态高度一致

在所有 8 组可视化实验中，Qwen2.5-3B（base）、Qwen2.5-3B-Instruct（aligned）、AZR-Coder-3B（self-evolved）的指标曲线几乎完全重合。说明：
- **逐层行为动态主要由预训练权重决定**，对齐训练和自进化没有从根本上改变内部信息流结构
- 不同训练范式改变的是模型的"输出决策"而非"内部识别"

#### 结论 2: 行为判别力随深度单调增长

- Probe accuracy 从 ~0.5（机会水平）稳步上升到 ~1.0
- 深层探针用更小的权重即可分类 → 行为信息在深层更容易被线性读出

#### 结论 3: Category Separation 与 Probe Accuracy 的悖论

- Category Separation 随深度递减（质心越来越近），但 Probe Accuracy 随深度递增
- 说明深层的行为差异编码在精细的线性方向中，而非粗粒度的几何分离

#### 结论 4: harmful 类别始终最容易分离

- 探针范数在所有层、所有模型中都是 harmful 最低
- 有害内容在激活空间中有非常清晰的信号，可能因为有害 prompt 有独特的词汇/语义模式
- 暗示：模型在预训练阶段就已经"知道"什么是有害的，对齐训练可能只是教它基于这个已有信号做出拒绝

#### 结论 5: 探针方向的三阶段渐变结构

- 热力图显示浅层（0-10）、中层（15-32）、末层（25-35）三个有重叠的聚集区域
- 不是离散的阶段跳变，而是连续渐进的方向旋转
- 可能对应：浅层（词汇/表层特征识别）→ 中层（语义理解重组）→ 深层（行为决策）

---

## 六、项目优点

### 6.1 架构设计

- **模块化清晰**: extractor → dynamics → behavior → cross_scale → visualizer 的管线设计层次分明
- **数据结构规范**: 大量使用 `@dataclass`，每个分析结果都有明确的类型定义
- **可扩展性好**: 支持多种模型架构，新增行为类别或分析指标都很方便
- **序列化完善**: report_io 支持新旧两种 JSON 格式 + npz 探针向量格式，方便复现

### 6.2 分析方法

- **多角度交叉验证**: 同一现象从 probe accuracy、direction cosine、BIC、bifurcation 等多个角度观察
- **统计严谨性**: 分叉分析使用 bootstrap CI，cross-scale 使用 Kendall tau 检验
- **BIC 指标设计**: Effective Rank + Explained Ratio + Inter-class Ratio 三合一，对行为信息浓度的刻画比单一指标更全面
- **全局 PCA 降维**: notebook 使用同一个 PCA 变换所有模型和层，确保不同模型在同一坐标系下可比

### 6.3 可视化

- **交互式 notebook**: 使用 ipywidgets + matplotlib widget 模式，支持滑块动态探索
- **8 组互补的可视化**: 从不同角度全面展示数据
- **3D 可视化**: 直观展示高维探针方向的几何关系

### 6.4 工程实践

- **内存管理**: 及时 `del extractor` + `torch.cuda.empty_cache()`
- **错误处理**: tokenizer 加载失败的 fallback，架构自动检测
- **Activation Steering Demo**: 提供了从分析到应用的完整闭环思路

---

## 七、项目不足与改进建议

### 7.1 实验设计层面

#### 7.1.1 样本量过少

- 每类仅 8 条 prompt，总共 32 条。在 2048 维空间中，LogisticRegression 的解高度不稳定
- **建议**: 每类至少 50-100 条，使用标准化数据集（如 AdvBench、TruthfulQA）

#### 7.1.2 行为类别太粗糙

- harmful / safe / factual / reasoning 是预训练阶段就能区分的表层语义类别
- 对齐训练改变的是"面对 harmful 时的应对策略"，不是"识别 harmful 的能力"
- **建议**: 增加更细粒度的标签（如 "harmful-refused" vs "harmful-complied"），捕捉模型行为决策的差异

#### 7.1.3 未区分模型实际输出行为

- 当前实验只看"模型内部能否区分行为类别"，不看"模型实际做了什么"
- base 模型对 harmful prompt 可能直接回答，而 instruct 模型可能拒绝——这个关键差异完全没有体现
- **建议**: 将分析与模型实际输出对齐，对比"回答了的 harmful" vs "拒绝了的 harmful"

#### 7.1.4 三个模型无法有效区分

- 8 组实验全部显示三个模型几乎一致，没有达到区分不同训练范式的目的
- **建议**: 设计针对性的对比实验——找 base 会回答但 instruct 会拒绝的 prompt，分析它们在激活空间的差异

### 7.2 方法论层面

#### 7.2.1 只分析了 residual stream

- 仅捕获了 decoder layer 输出（residual stream），未分析 attention pattern、MLP 输入输出、特定 attention head
- 对齐训练可能更多地改变了特定组件（如某些 head 的注意力模式），而非整体 residual stream
- **建议**: 增加 attention head 级别的分析，参考 Anthropic 的 "A Mathematical Framework for Transformer Circuits"

#### 7.2.2 探针方法的局限

- LogisticRegression 是线性探针，只能检测线性可分的信息
- 如果行为信息以非线性方式编码，线性探针会低估模型的内部区分能力
- **建议**: 补充非线性探针（如小型 MLP 探针），或使用 CKA/CCA 等不依赖分类器的方法

#### 7.2.3 只看了最后一个 token

- 所有分析都基于最后一个 token 的 residual stream（`token_idx=-1`）
- 对于较长的 prompt，中间 token 的信息可能也很重要
- **建议**: 对比不同 token 位置的分析结果，或使用 mean pooling

### 7.3 工程层面

#### 7.3.1 Phase 2 未执行

- 跨规模分析（Critical Window Hypothesis）是三个研究问题之一，但从未实际运行
- `cross_scale.py` 代码完整，但 results 目录中没有相关输出

#### 7.3.2 Demo 未集成

- Activation Steering demo 是独立的 git 子仓库，与主框架没有代码级集成
- 理想的工作流应该是：llm_lens 分析 → 定位关键层和方向 → activation steering 验证

#### 7.3.3 结果复现性

- 多次运行的结果（20260414 × 3, 20260422 × 1）之间的差异未被分析
- npz 文件的命名不够规范，notebook 中硬编码了具体路径

#### 7.3.4 缺少自动化评估

- 没有定量评估不同模型之间差异的统计检验
- 所有结论都基于视觉观察，缺少 p-value 或效应量
- **建议**: 对三个模型的各指标做配对检验（如 Wilcoxon signed-rank test），量化差异的统计显著性

### 7.4 叙事与呈现层面

#### 7.4.1 发现与预期的偏差

- 项目标题 "The Inner of Self-Evolving Agents" 暗示重点是研究自进化模型的内部机制
- 但实际结果是三种训练范式的内部行为动态几乎无差异
- 这个"空结果"本身有价值（说明行为动态由预训练决定），但需要更好的叙述

#### 7.4.2 缺少论文级的实验报告

- 大量分析结果停留在 notebook 级别的探索性可视化
- 缺少系统的实验报告，没有 ablation study 或假设检验

---

## 八、后续工作建议

### 8.1 短期（增强现有框架的区分能力）

1. **扩大样本量**: 每类 50+ prompt，使用标准 benchmark 数据集
2. **添加行为决策标签**: 收集三个模型对每条 prompt 的实际输出，区分 "回答" vs "拒绝"
3. **执行 Phase 2**: 在 7B/14B 模型上运行，实际检验 Critical Window Hypothesis
4. **添加统计检验**: 对三个模型的指标差异做显著性检验

### 8.2 中期（深化分析方法）

5. **组件级分析**: 拆解到 attention head、MLP gate/up/down、LayerNorm 等子组件
6. **非线性探针**: 补充 MLP 探针或 CKA 分析
7. **激活差异分析**: 直接比较同一 prompt 在三个模型中的激活值差异，而非只比较探针
8. **集成 Activation Steering**: 用 llm_lens 定位关键层 → 用 steering 验证因果关系

### 8.3 长期（研究问题的深化）

9. **因果干预**: 在关键层 ablate 或注入特定方向，观察输出行为变化
10. **自进化过程追踪**: 如果可以获得 AZR-Coder 的训练 checkpoint，追踪自进化过程中内部表示的变化
11. **多模型家族对比**: 不仅限于 Qwen 系列，扩展到 Llama、Mistral 等不同模型家族

---

## 九、总结

本项目构建了一个完整的 LLM 逐层行为动态分析框架，代码架构清晰、分析方法多维。其最重要的发现是：**三种截然不同的训练范式（预训练、对齐、自进化）在内部行为判别动态上几乎没有差异**，这暗示大语言模型的行为认知能力主要在预训练阶段形成，后续训练改变的只是"知道之后怎么做"而非"能不能知道"。

然而，由于样本量不足、行为类别过粗、分析仅限于 residual stream 的线性探针，现有实验可能低估了模型间的真实差异。项目需要在实验设计和方法论上进行针对性强化，才能更有力地支撑或推翻其核心假设。
