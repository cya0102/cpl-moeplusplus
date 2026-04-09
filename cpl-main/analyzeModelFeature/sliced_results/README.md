# 切片分析结果说明文档

本文档详细解释 `sliced_analysis.py` 和 `visualize_sliced.py` 生成的所有输出文件的含义，包括核心概念、每个字段的意义、以及每张图表的阅读方法。

---

## 目录

1. [核心概念](#1-核心概念)
2. [输出文件总览](#2-输出文件总览)
3. [analysis_report.txt 详解](#3-analysis_reporttxt-详解)
4. [analysis_data.json 详解](#4-analysis_datajson-详解)
5. [图表详解](#5-图表详解)
6. [所有维度与切片的完整含义](#6-所有维度与切片的完整含义)
7. [评估指标说明](#7-评估指标说明)

---

## 1. 核心概念

### 1.1 什么是"维度"（Dimension）

**维度**是我们用来"切分"数据集的**分类标准**。

由于 Video-Text Grounding 数据集（Charades-STA、ActivityNet Captions）没有显式的类别标签（如"跑步"、"做饭"），我们无法直接按类别分组分析。因此，脚本从每个样本的**描述文本**和**时序区间**中自动提取"隐式属性"，每种属性就是一个"维度"。

**类比**：假设你有一份学生成绩表，你想分析"哪些学生考得好"。你可以按"性别"分、按"年级"分、按"城市/农村"分——这里的"性别"、"年级"、"城乡"就是不同的**维度**。

### 1.2 什么是"切片"（Slice）

**切片**是某个维度下的**一个具体取值/分桶**。

例如，维度 `query_length_bucket`（查询文本长度）下有三个切片：
- `short(<=5)` — 文本长度 ≤ 5 个单词的样本
- `medium(6-10)` — 文本长度在 6~10 个单词的样本
- `long(11-15)` — 文本长度在 11~15 个单词的样本

**类比**：维度"年级"下的切片就是"一年级"、"二年级"、"三年级"。

### 1.3 分析的核心逻辑

整个分析的思路是：

```
对于每个维度（如"查询文本长度"）：
    将所有样本按该维度的取值分成若干切片（如 short / medium / long）
    对每个切片：
        分别计算各模型（CPL_Fusion、CPL_DMVS 等）的性能指标
    对比各模型在不同切片上的表现差异
```

这样我们就能回答："模型 A 擅长处理什么类型的样本？在什么类型的样本上不如模型 B？"

---

## 2. 输出文件总览

```
sliced_results/
├── charades/                          # Charades-STA 数据集的结果
│   ├── analysis_report.txt            # 纯文本报告（人类可读的表格）
│   ├── analysis_data.json             # JSON 格式的结构化数据（供程序读取或进一步分析）
│   └── figures/                       # 可视化图表
│       ├── bar_*.png                  # 柱状对比图
│       ├── radar_*.png                # 雷达图
│       ├── heatmap_divergence.png     # 差异热力图
│       ├── scatter_comparison.png     # 散点对比图
│       └── waterfall_gain.png         # 瀑布增益图
└── activitynet/                       # ActivityNet Captions 数据集的结果
    ├── analysis_report.txt
    ├── analysis_data.json
    └── figures/
        └── ...（同上）
```

---

## 3. analysis_report.txt 详解

这是纯文本格式的分析报告，用表格形式展示所有结果。下面逐段解释。

### 3.1 头部信息

```
数据集: charades
模型: CPL_Fusion, CPL_DMVS
总样本数: 3158
IoU 阈值: [0.3, 0.5, 0.7]
```

| 字段 | 含义 |
|------|------|
| 数据集 | 分析的是哪个数据集。`charades` = Charades-STA，`activitynet` = ActivityNet Captions |
| 模型 | 参与对比的模型名称列表 |
| 总样本数 | 测试集中的样本总数（Charades-STA 有 3158 个测试样本，ActivityNet 有 17031 个） |
| IoU 阈值 | 用于计算 R@1 指标的 IoU 门槛。Charades-STA 使用 0.3/0.5/0.7，ActivityNet 使用 0.1/0.3/0.5 |

### 3.2 全局指标

```
CPL_Fusion:
    R@1,IoU=0.3: 44.65%
    R@1,IoU=0.5: 27.68%
    R@1,IoU=0.7: 8.01%
    mIoU: 28.60%
```

这是**不做任何切片**、在整个测试集上计算的总体指标。相当于模型的"总成绩"。各指标含义详见[第 7 节](#7-评估指标说明)。

### 3.3 维度表格

```
================================================================================
  维度: query_length_bucket  (共 3 个切片)
================================================================================
切片                  数量  Fusion_0.3  Fusion_0.5  Fusion_0.7  Fusion_mIoU  DMVS_0.3  DMVS_0.5  DMVS_0.7  DMVS_mIoU
---------------------------------------------------------------------------------------------------------------------
medium(6-10)         1914      46.3       28.0        8.5        29.6       37.1       24.0        9.8       24.4
short(<=5)           1243      42.0       27.1        7.2        27.0       38.9       25.3       10.4       25.6
long(11-15)             1     100.0      100.0        0.0        53.9        0.0        0.0        0.0       17.5
```

**表头含义**：
| 列名 | 含义 |
|------|------|
| 切片 | 该维度下的具体分组名称及其分桶范围 |
| 数量 | 该切片包含多少个样本 |
| `Fusion_0.3` | CPL_Fusion 模型在该切片上的 R@1,IoU=0.3 指标（百分制） |
| `Fusion_0.5` | CPL_Fusion 模型在该切片上的 R@1,IoU=0.5 指标（百分制） |
| `Fusion_0.7` | CPL_Fusion 模型在该切片上的 R@1,IoU=0.7 指标（百分制） |
| `Fusion_mIoU` | CPL_Fusion 模型在该切片上的平均 IoU（百分制） |
| `DMVS_0.3` 等 | CPL_DMVS 模型的对应指标 |

**怎么读这个表**：以 `medium(6-10)` 这一行为例——
> 在 Charades-STA 测试集中，有 1914 个样本的查询文本长度在 6~10 个单词之间。
> 在这 1914 个样本上，CPL_Fusion 的 R@1,IoU=0.5 是 28.0%，CPL_DMVS 的是 24.0%。
> 说明在"中等长度查询"这类样本上，CPL_Fusion 比 CPL_DMVS 表现更好（高 4 个百分点）。

### 3.4 模型间差异最大的切片

```
#1  [ends_at_end] True  (n=103, 差异=34.0pp)
     最优: CPL_DMVS  |  最差: CPL_Fusion
       CPL_Fusion: 6.8%
       CPL_DMVS: 40.8%
```

| 字段 | 含义 |
|------|------|
| `[ends_at_end]` | 维度名称 |
| `True` | 切片取值（GT 时序区间结束于视频末尾的样本） |
| `n=103` | 该切片有 103 个样本 |
| `差异=34.0pp` | 两个模型在该切片上的 R@1,IoU=0.5 差值为 34.0 个百分点（pp = percentage point） |
| 最优/最差 | 在该切片上哪个模型最好、哪个最差 |

**含义**：这部分自动找出模型之间"差距最大"的地方。上面的例子说明：当 GT 片段恰好在视频末尾时，CPL_DMVS（40.8%）远远好于 CPL_Fusion（6.8%）。

### 3.5 各模型优劣势分析

```
--- CPL_Fusion ---
  擅长 (Top-5):
    [gt_coverage_bucket] medium(30-50%)  R@1,IoU=0.5=46.6%  (n=1068)
  不擅长 (Bottom-5):
    [gt_coverage_bucket] tiny(<=10%)  R@1,IoU=0.5=0.0%  (n=24)
```

分别列出每个模型**表现最好**和**表现最差**的 5 个切片，帮助你快速了解每个模型的特长和短板。

### 3.6 交叉切片分析

交叉切片 = 同时按两个维度切分。例如 `query_length_bucket × gt_coverage_bucket` 会产生如 `medium(6-10) × small(10-30%)` 这样的组合切片，用于发现更细粒度的模式。

---

## 4. analysis_data.json 详解

这是 txt 报告的结构化版本，适合程序读取或后续自定义分析。

### 4.1 顶层结构

```json
{
  "dataset": "charades",
  "model_names": ["CPL_Fusion", "CPL_DMVS"],
  "num_samples": 3158,
  "iou_thresholds": [0.3, 0.5, 0.7],
  "dimension_results": { ... },
  "divergences": [ ... ],
  "strengths": { ... }
}
```

| 字段 | 类型 | 含义 |
|------|------|------|
| `dataset` | 字符串 | 数据集名称 |
| `model_names` | 字符串列表 | 参与对比的模型名称 |
| `num_samples` | 整数 | 测试集总样本数 |
| `iou_thresholds` | 浮点数列表 | 使用的 IoU 阈值 |
| `dimension_results` | 对象 | **核心内容**：每个维度的切片分析结果 |
| `divergences` | 列表 | 模型差异最大的切片排行榜 |
| `strengths` | 对象 | 每个模型的优劣势切片 |

### 4.2 `dimension_results` 详解

这是最重要的字段。它是一个字典，**键**是维度名称，**值**是该维度的分析结果。

```json
"dimension_results": {
  "query_length_bucket": { ... },
  "action_complexity": { ... },
  "action_category": { ... },
  "gt_coverage_bucket": { ... },
  ...
}
```

每个维度名称的含义见[第 6 节](#6-所有维度与切片的完整含义)。

#### 单个维度的结构（以 `query_length_bucket` 为例）

```json
"query_length_bucket": {
  "dimension": "query_length_bucket",
  "num_slices": 3,
  "slices": {
    "short(<=5)": { ... },
    "medium(6-10)": { ... },
    "long(11-15)": { ... }
  }
}
```

| 字段 | 含义 |
|------|------|
| `dimension` | 维度名称（与键名相同，冗余保留方便独立使用） |
| `num_slices` | 该维度下有多少个切片 |
| `slices` | 字典，每个键是切片名称，值是该切片的详细数据 |

#### 单个切片的结构（以 `"medium(6-10)"` 为例）

```json
"medium(6-10)": {
  "count": 1914,
  "ratio": 0.606,
  "models": {
    "CPL_Fusion": {
      "R@1,IoU=0.3": 0.463,
      "R@1,IoU=0.5": 0.280,
      "R@1,IoU=0.7": 0.085,
      "mIoU": 0.296,
      "count": 1914,
      "R@5,mIoU": 0.531,
      "R@5,IoU=0.3": 0.965,
      "R@5,IoU=0.5": 0.823,
      "R@5,IoU=0.7": 0.442
    },
    "CPL_DMVS": {
      "R@1,IoU=0.3": 0.371,
      ...
    }
  }
}
```

| 字段 | 含义 |
|------|------|
| `count` | 该切片包含的样本数（1914 个样本的查询文本长度在 6~10 个单词） |
| `ratio` | 该切片占总样本的比例（1914/3158 ≈ 0.606，即 60.6% 的样本属于此切片） |
| `models` | 字典，每个模型在该切片上的各项指标 |
| `R@1,IoU=0.3` | 值为 0~1 的浮点数。0.463 表示 46.3%。含义：在这 1914 个样本中，模型排名第 1 的预测与 GT 的 IoU ≥ 0.3 的比例 |
| `R@1,IoU=0.5` | 同上，但 IoU 门槛为 0.5（更严格） |
| `R@1,IoU=0.7` | 同上，IoU 门槛为 0.7（最严格） |
| `mIoU` | 平均 IoU：所有样本的 top-1 预测与 GT 的 IoU 的算术平均值 |
| `R@5,IoU=0.3` | 在排名前 5 的预测中，只要有一个 IoU ≥ 0.3 就算正确的比例 |
| `R@5,mIoU` | 排名前 5 中最好预测的平均 IoU |

> **注意**：JSON 中的指标值是 0~1 的小数（如 0.463），txt 报告中是百分制（如 46.3%）。

### 4.3 `divergences` 详解

```json
"divergences": [
  {
    "dimension": "ends_at_end",
    "slice": "True",
    "count": 103,
    "metric": "R@1,IoU=0.5",
    "max_diff": 0.3398,
    "best_model": "CPL_DMVS",
    "worst_model": "CPL_Fusion",
    "model_values": {
      "CPL_Fusion": 0.0680,
      "CPL_DMVS": 0.4078
    }
  },
  ...
]
```

这是一个列表，按 `max_diff`（差异大小）从大到小排列，最多保留 30 条。

| 字段 | 含义 |
|------|------|
| `dimension` | 这个差异出现在哪个维度 |
| `slice` | 具体是哪个切片 |
| `count` | 该切片的样本数 |
| `metric` | 用来衡量差异的指标（默认为中间 IoU 阈值的 R@1） |
| `max_diff` | 模型间的最大差异（0.3398 = 33.98 个百分点） |
| `best_model` | 在该切片上表现最好的模型 |
| `worst_model` | 在该切片上表现最差的模型 |
| `model_values` | 每个模型在该切片上的指标值 |

### 4.4 `strengths` 详解

```json
"strengths": {
  "CPL_Fusion": {
    "best_slices": [
      {
        "dimension": "gt_coverage_bucket",
        "slice": "medium(30-50%)",
        "count": 1068,
        "R@1,IoU=0.5": 0.4663
      },
      ...
    ],
    "worst_slices": [
      {
        "dimension": "gt_coverage_bucket",
        "slice": "tiny(<=10%)",
        "count": 24,
        "R@1,IoU=0.5": 0.0
      },
      ...
    ]
  },
  "CPL_DMVS": { ... }
}
```

| 字段 | 含义 |
|------|------|
| `best_slices` | 该模型得分最高的 10 个切片（按 R@1,IoU=0.5 降序） |
| `worst_slices` | 该模型得分最低的 10 个切片（按 R@1,IoU=0.5 升序） |

---

## 5. 图表详解

### 5.1 柱状对比图 `bar_*.png`

每个柱状图对应一个维度。X 轴是该维度的各个切片，Y 轴是指标值（百分制）。不同颜色的柱子代表不同模型。

| 文件名 | 对应维度 | 使用指标 | 含义 |
|--------|----------|----------|------|
| `bar_query_length_bucket.png` | 查询文本长度 | R@1,IoU=中间阈值 | 对比模型在不同长度查询上的表现 |
| `bar_query_length_bucket_miou.png` | 查询文本长度 | mIoU | 同上，但用 mIoU 指标 |
| `bar_action_complexity.png` | 动作复杂度 | R@1,IoU=中间阈值 | 对比模型在单动作/多动作查询上的表现 |
| `bar_action_complexity_miou.png` | 动作复杂度 | mIoU | 同上 |
| `bar_gt_coverage_bucket.png` | GT 覆盖率 | R@1,IoU=中间阈值 | 对比模型在不同长度占比的片段上的表现 |
| `bar_gt_coverage_bucket_miou.png` | GT 覆盖率 | mIoU | 同上 |
| `bar_gt_position_bucket.png` | GT 位置 | R@1,IoU=中间阈值 | 对比模型在视频开头/中间/结尾片段上的表现 |
| `bar_gt_position_bucket_miou.png` | GT 位置 | mIoU | 同上 |
| `bar_gt_length_bucket.png` | GT 绝对长度 | R@1,IoU=中间阈值 | 对比模型在不同绝对时长片段上的表现 |
| `bar_gt_length_bucket_miou.png` | GT 绝对长度 | mIoU | 同上 |
| `bar_video_length_bucket.png` | 视频长度 | R@1,IoU=中间阈值 | 对比模型在不同时长视频上的表现 |
| `bar_video_length_bucket_miou.png` | 视频长度 | mIoU | 同上 |
| `bar_query_density_bucket.png` | 查询密度 | R@1,IoU=中间阈值 | 对比模型在查询密度不同的视频上的表现 |
| `bar_query_density_bucket_miou.png` | 查询密度 | mIoU | 同上 |
| `bar_coverage_R@1_IoU0.3.png` | GT 覆盖率 | R@1,IoU=0.3 | 针对特定 IoU 阈值的覆盖率分析 |
| `bar_coverage_R@1_IoU0.5.png` | GT 覆盖率 | R@1,IoU=0.5 | 同上 |
| `bar_coverage_R@1_IoU0.7.png` | GT 覆盖率 | R@1,IoU=0.7 | 同上 |

**阅读方法**：
- 柱子越高 = 该模型在该切片上的性能越好
- 同一个切片内，比较不同颜色柱子的高度差 = 模型间的差异
- 每根柱子上方标注了具体数值
- X 轴标签下方的 `(n=xxx)` 表示该切片的样本数。样本数越大，结果越可靠

### 5.2 雷达图 `radar_*.png`

雷达图用多边形展示各模型在同一维度的不同切片上的表现。每个顶点是一个切片，线条越外圈 = 性能越好。

| 文件名 | 对应维度 |
|--------|----------|
| `radar_query_length_bucket.png` | 查询文本长度（仅在切片 ≥ 3 时生成） |
| `radar_action_complexity.png` | 动作复杂度 |
| `radar_gt_coverage_bucket.png` | GT 覆盖率 |
| `radar_gt_position_bucket.png` | GT 位置 |
| `radar_gt_length_bucket.png` | GT 绝对长度 |
| `radar_video_length_bucket.png` | 视频长度 |
| `radar_query_density_bucket.png` | 查询密度 |

**阅读方法**：
- 如果模型 A 的多边形在某个方向比模型 B 突出更多，说明模型 A 在那个切片上更强
- 多边形越"圆"说明模型表现越均衡，越"尖"说明模型在某些类型上有明显的强弱差异

### 5.3 差异热力图 `heatmap_divergence.png`

这张图展示**模型间差异最大的 Top-25 个切片**。

- 每一**行**是一个切片（格式为 `[维度名] 切片名 (n=样本数)`）
- 每一**列**是一个模型
- 每个单元格的**颜色深浅**表示该模型在该切片上的指标值（越深 = 越高）
- 每个单元格内标注了**具体数值**

**阅读方法**：
- 同一行内，如果两个模型的数值/颜色差异很大，说明该切片是模型差异的关键点
- 行是按差异从大到小排列的，最上面的行是差异最显著的切片

### 5.4 散点对比图 `scatter_comparison.png`

这张图用于直观对比两个模型。

- **X 轴** = 模型 A 在某个切片上的指标
- **Y 轴** = 模型 B 在同一个切片上的指标
- 每个**点**代表一个切片
- 点的**大小** = 该切片的样本数（越大越可靠）
- 图中有一条**对角线**（虚线）

**阅读方法**：
- 点落在**对角线上方** → 模型 B 在该切片上更好
- 点落在**对角线下方** → 模型 A 在该切片上更好
- 点离对角线**越远** → 两个模型的差异越大
- 离对角线最远的 5 个点会被标注出具体的维度和切片名

### 5.5 瀑布增益图 `waterfall_gain.png`

这张图展示模型 B 相对于模型 A 的**增益或退化**。

- 每个**横条**代表一个切片
- 横条向**右**延伸（绿色）= 模型 B 比模型 A **更好**，数值为领先多少个百分点
- 横条向**左**延伸（红色）= 模型 B 比模型 A **更差**，数值为落后多少个百分点
- 只展示差异最大的各 10 个切片（最好的 10 个和最差的 10 个）

**阅读方法**：
- 看绿色条：了解模型 B 的优势场景
- 看红色条：了解模型 B 的劣势场景

---

## 6. 所有维度与切片的完整含义

### 6.1 文本属性维度

这些维度通过分析查询文本（如 `"a person is putting a book on a shelf."`）的词汇内容自动提取。

#### `query_length_bucket`（查询文本长度）

将查询文本按单词数量分桶。

| 切片名 | 含义 | 举例 |
|--------|------|------|
| `short(<=5)` | 不超过 5 个单词 | `"person turn a light on."` (5 词) |
| `medium(6-10)` | 6~10 个单词 | `"a person is putting a book on a shelf."` (9 词) |
| `long(11-15)` | 11~15 个单词 | `"the man walks over to the table and picks up a glass"` |
| `very_long(>15)` | 超过 15 个单词 | ActivityNet 中较长的描述 |

**分析意义**：查询越长，通常包含更多信息，但也更难建模。可以观察模型对长/短查询的处理能力差异。

#### `action_complexity`（动作复杂度）

通过计算查询文本中**动词的数量**来衡量描述的复杂度。

| 切片名 | 含义 | 举例 |
|--------|------|------|
| `no_verb` | 没有识别到动词 | `"the man in the red shirt"` |
| `single_action` | 只有 1 个动词 | `"a person is walking."` (walking) |
| `dual_action` | 有 2 个动词 | `"person opens the door and walks in."` (opens, walks) |
| `multi_action(3+)` | 有 3 个或更多动词 | `"she picks up the bag, walks to the door and opens it."` |

**分析意义**：多动作描述意味着被定位的视频片段中包含多个连续动作，通常更难定位。

#### `action_category`（动作类别）

通过词表匹配，将查询中的动词归入 4 大类别。一个查询可能属于多个类别。

| 类别名 | 含义 | 包含的典型动词 |
|--------|------|----------------|
| `movement` | 运动/位移类 | walk, run, jump, sit, stand, climb, swim, ride, enter, leave |
| `interaction` | 交互/操作类 | pick, put, take, hold, throw, catch, push, pull, grab, carry, pour, cut |
| `state_change` | 状态变化类 | open, close, turn, flip, start, stop, break, fix, wear, light, switch |
| `perception` | 感知/消费类 | look, watch, read, talk, speak, sing, listen, laugh, play, eat, drink, cook |
| `unknown` | 未识别 | 没有匹配到任何预定义动词的查询 |
| `movement,perception` | 同时包含两类 | `"person walks over and talks to someone"` |

**分析意义**：不同类别的动作在视觉特征上差异很大。movement 类通常有明显的运动特征，而 perception 类（如 look、read）的视觉变化可能很微妙。

#### `has_person_subject`（是否有人称主语）

查询文本中是否包含人称词汇。

| 切片名 | 含义 |
|--------|------|
| `True` | 包含 person、man、woman、he、she、they、someone 等 |
| `False` | 不包含人称词汇（如 `"the ball rolls off the table"`） |

#### `has_temporal_cue`（是否有时序线索词）

查询文本中是否包含时间/顺序相关的词汇。

| 切片名 | 含义 | 时序线索词示例 |
|--------|------|----------------|
| `True` | 包含时序词 | then, after, before, while, during, first, next, again, continue, still |
| `False` | 不包含时序词 | 大多数样本 |

**分析意义**：包含时序线索词的查询往往描述的是一系列事件中的某一段，可能需要模型理解事件的先后关系。

### 6.2 时序属性维度

这些维度通过分析 GT（Ground Truth，即标注的正确时序区间）和视频长度自动计算。

#### `gt_coverage_bucket`（GT 覆盖率）

GT 片段长度占整个视频长度的比例。

| 切片名 | 含义 | 示例 |
|--------|------|------|
| `tiny(<=10%)` | GT 不到视频的 10% | 30 秒视频中定位一个 2 秒的片段 |
| `small(10-30%)` | GT 占视频 10%~30% | 30 秒视频中定位一个 6 秒的片段 |
| `medium(30-50%)` | GT 占视频 30%~50% | 30 秒视频中定位一个 12 秒的片段 |
| `large(50-70%)` | GT 占视频 50%~70% | 30 秒视频中定位一个 18 秒的片段 |
| `dominant(>70%)` | GT 占视频 70% 以上 | 几乎等于整个视频 |

**分析意义**：这是**最重要的维度之一**。覆盖率越小，定位越困难（大海捞针）；覆盖率越大，越容易碰巧命中。

#### `gt_position_bucket`（GT 在视频中的位置）

GT 片段的中心点在视频中的相对位置。

| 切片名 | 含义 |
|--------|------|
| `beginning` | GT 中心点在视频的前 1/3 |
| `middle` | GT 中心点在视频的中间 1/3 |
| `end` | GT 中心点在视频的后 1/3 |

**分析意义**：某些模型可能存在位置偏差（position bias），比如倾向于预测视频开头的片段。

#### `video_length_bucket`（视频长度）

视频的总时长分桶。

| 切片名 | 含义 |
|--------|------|
| `very_short(<=20s)` | ≤ 20 秒 |
| `short(20-40s)` | 20~40 秒 |
| `medium(40-80s)` | 40~80 秒 |
| `long(80-160s)` | 80~160 秒 |
| `very_long(>160s)` | > 160 秒 |

**分析意义**：视频越长，候选区间越多，定位越困难。

#### `gt_length_bucket`（GT 绝对长度）

GT 片段的绝对时长分桶。

| 切片名 | 含义 |
|--------|------|
| `very_short(<=3s)` | ≤ 3 秒 |
| `short(3-8s)` | 3~8 秒 |
| `medium(8-20s)` | 8~20 秒 |
| `long(20-60s)` | 20~60 秒 |
| `very_long(>60s)` | > 60 秒 |

#### `starts_at_beginning`（是否起始于视频开头）

GT 片段的起始时间是否在视频的前 1 秒内。

| 切片名 | 含义 |
|--------|------|
| `True` | GT 从视频开头（< 1 秒）开始 |
| `False` | GT 不从视频开头开始 |

#### `ends_at_end`（是否结束于视频结尾）

GT 片段的结束时间是否在视频结尾的最后 1 秒内。

| 切片名 | 含义 |
|--------|------|
| `True` | GT 到视频末尾（距结尾 < 1 秒）结束 |
| `False` | GT 不在视频末尾结束 |

**分析意义**：视频边界处的片段可能有特殊性，例如模型可能难以正确预测延伸到视频最后一帧的片段。

### 6.3 多查询属性维度

这些维度分析的是同一视频内多个查询之间的关系。

#### `query_density_bucket`（查询密度）

同一个视频中有多少个不同的查询样本。

| 切片名 | 含义 |
|--------|------|
| `single_query` | 该视频只有 1 个查询 |
| `sparse(2-3)` | 该视频有 2~3 个查询 |
| `medium(4-6)` | 该视频有 4~6 个查询 |
| `dense(7+)` | 该视频有 7 个或更多查询 |

**分析意义**：查询密集的视频通常内容更复杂，包含多个可区分的事件，模型需要更精细地理解文本差异。

#### `has_overlapping_queries`（是否有时序重叠的查询）

当前样本的 GT 区间是否与同一视频中其他样本的 GT 区间有重叠。

| 切片名 | 含义 |
|--------|------|
| `True` | 存在重叠（如两个不同的查询描述同一段视频中不同方面的事情） |
| `False` | 不存在重叠 |

**分析意义**：重叠查询意味着不同描述指向视频中相近甚至相同的时间段，模型需要能区分微妙的文本差异。

---

## 7. 评估指标说明

### R@1,IoU=θ

全称 **Recall at rank 1, IoU threshold θ**。

计算方法：
1. 模型为每个样本生成多个预测片段（proposals），按打分从高到低排序
2. 取排名**第 1**的预测片段
3. 计算该预测片段与 GT（真实标注）之间的 IoU（交并比）
4. 如果 IoU ≥ θ，则该样本"命中"
5. R@1,IoU=θ = 命中样本数 / 总样本数

**IoU（Intersection over Union，交并比）**的计算：

```
           |----GT----|
                |----Pred----|

交集 = 重叠部分的长度
并集 = GT长度 + Pred长度 - 交集
IoU  = 交集 / 并集
```

| IoU 阈值 | 严格程度 | 含义 |
|----------|----------|------|
| 0.1 | 非常宽松 | 只要有 10% 的重叠就算对 |
| 0.3 | 宽松 | 需要 30% 的重叠 |
| 0.5 | 中等 | 需要 50% 的重叠（最常用的标准） |
| 0.7 | 严格 | 需要 70% 的重叠（要求几乎精确定位） |

### R@5,IoU=θ

与 R@1 类似，但取排名**前 5**的预测片段，只要其中**任意一个** IoU ≥ θ 就算命中。

### mIoU

**Mean IoU**：所有样本的 top-1 预测与 GT 的 IoU 的算术平均值。不设阈值，直接反映平均重叠程度。

---

## 8. 快速阅读指南

如果你只想快速了解结果，推荐按以下顺序查看：

1. **先看 `analysis_report.txt` 的"全局指标"部分**，了解各模型的整体性能排名
2. **再看"模型间差异最大的切片"部分**，找到模型差异最显著的场景
3. **看 `waterfall_gain.png`**，直观了解模型 B 相对于 A 在哪些场景上有增益/退化
4. **看 `heatmap_divergence.png`**，一目了然地看到差异最大的切片
5. **按需查看各维度的 `bar_*.png`**，深入了解某个具体维度上的表现差异
6. 如果需要做进一步的自定义分析，读取 `analysis_data.json`
