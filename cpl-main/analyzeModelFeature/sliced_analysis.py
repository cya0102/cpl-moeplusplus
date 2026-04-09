"""
细粒度切片分析脚本 —— 通过提取隐式属性对数据集进行切片，
对比多个模型在不同子集上的表现差异。

用法:
    python sliced_analysis.py \
        --dataset charades \
        --pred-files CharadesSTA_CPL.json CharadesSTA_CPL_MoE.json CharadesSTA_CPL_DMVS.json \
        --output-dir sliced_results/charades

    python sliced_analysis.py \
        --dataset activitynet \
        --pred-files ActivityNet_CPL.json ActivityNet_CPL_MoE.json ActivityNet_CPL_DMVS.json \
        --output-dir sliced_results/activitynet
"""

import os
import sys
import json
import re
import csv
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

# ──────────────────── 加载数据集注释 ────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

DATASET_ANNO_MAP = {
    "charades": os.path.join(DATA_DIR, "charades", "test.json"),
    "activitynet": os.path.join(DATA_DIR, "activitynet", "test_data.json"),
}

DATASET_IOU_THRESHOLDS = {
    "charades": [0.3, 0.5, 0.7],
    "activitynet": [0.1, 0.3, 0.5],
}


def load_annotations(dataset: str) -> List[List]:
    path = DATASET_ANNO_MAP[dataset]
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_predictions(pred_file: str) -> Dict:
    with open(pred_file, "r", encoding="utf-8") as f:
        return json.load(f)


# ──────────────────── 隐式属性提取 ────────────────────

# ── 动词词表（常见动作类别） ──
ACTION_VERBS = {
    "movement": {
        "walk", "walks", "walking", "run", "runs", "running",
        "jump", "jumps", "jumping", "stand", "stands", "standing",
        "sit", "sits", "sitting", "climb", "climbs", "climbing",
        "crawl", "crawls", "crawling", "move", "moves", "moving",
        "go", "goes", "going", "come", "comes", "coming",
        "enter", "enters", "entering", "leave", "leaves", "leaving",
        "step", "steps", "stepping", "dance", "dances", "dancing",
        "swim", "swims", "swimming", "dive", "dives", "diving",
        "ride", "rides", "riding", "fly", "flies", "flying",
        "slide", "slides", "sliding", "roll", "rolls", "rolling",
    },
    "interaction": {
        "pick", "picks", "picking", "put", "puts", "putting",
        "take", "takes", "taking", "hold", "holds", "holding",
        "throw", "throws", "throwing", "catch", "catches", "catching",
        "push", "pushes", "pushing", "pull", "pulls", "pulling",
        "grab", "grabs", "grabbing", "drop", "drops", "dropping",
        "carry", "carries", "carrying", "lift", "lifts", "lifting",
        "give", "gives", "giving", "hand", "hands", "handing",
        "hit", "hits", "hitting", "kick", "kicks", "kicking",
        "touch", "touches", "touching", "hug", "hugs", "hugging",
        "shake", "shakes", "shaking", "pour", "pours", "pouring",
        "cut", "cuts", "cutting", "stir", "stirs", "stirring",
    },
    "state_change": {
        "open", "opens", "opening", "close", "closes", "closing",
        "turn", "turns", "turning", "flip", "flips", "flipping",
        "start", "starts", "starting", "stop", "stops", "stopping",
        "begin", "begins", "beginning", "finish", "finishes", "finishing",
        "break", "breaks", "breaking", "fix", "fixes", "fixing",
        "light", "lights", "lighting", "switch",
        "wear", "wears", "wearing", "dress", "dresses", "dressing",
        "undress", "wrap", "wraps", "wrapping",
    },
    "perception": {
        "look", "looks", "looking", "watch", "watches", "watching",
        "see", "sees", "seeing", "read", "reads", "reading",
        "talk", "talks", "talking", "speak", "speaks", "speaking",
        "say", "says", "saying", "sing", "sings", "singing",
        "listen", "listens", "listening", "laugh", "laughs", "laughing",
        "cry", "cries", "crying", "smile", "smiles", "smiling",
        "play", "plays", "playing", "eat", "eats", "eating",
        "drink", "drinks", "drinking", "cook", "cooks", "cooking",
    },
}

# 把所有类别的动词合并成一个集合，用于快速查找
ALL_ACTION_VERBS = set()
for verbs in ACTION_VERBS.values():
    ALL_ACTION_VERBS.update(verbs)


def _tokenize(text: str) -> List[str]:
    """简单分词：小写 + 按非字母分割"""
    return re.findall(r"[a-z]+", text.lower())


def extract_text_attributes(query: str) -> Dict[str, Any]:
    """从描述文本中提取隐式属性"""
    tokens = _tokenize(query)
    num_words = len(tokens)

    # 1. 查询长度分桶
    if num_words <= 5:
        length_bucket = "short(<=5)"
    elif num_words <= 10:
        length_bucket = "medium(6-10)"
    elif num_words <= 15:
        length_bucket = "long(11-15)"
    else:
        length_bucket = "very_long(>15)"

    # 2. 动作类别
    verb_categories = set()
    verbs_found = []
    for token in tokens:
        if token in ALL_ACTION_VERBS:
            verbs_found.append(token)
            for cat, verb_set in ACTION_VERBS.items():
                if token in verb_set:
                    verb_categories.add(cat)

    num_verbs = len(verbs_found)
    action_category = sorted(verb_categories) if verb_categories else ["unknown"]

    # 3. 动作复杂度（通过动词数量衡量）
    if num_verbs == 0:
        action_complexity = "no_verb"
    elif num_verbs == 1:
        action_complexity = "single_action"
    elif num_verbs == 2:
        action_complexity = "dual_action"
    else:
        action_complexity = "multi_action(3+)"

    # 4. 是否包含人称主语
    person_words = {"person", "people", "man", "woman", "boy", "girl", "child",
                    "he", "she", "they", "someone", "somebody", "lady",
                    "guy", "kid", "baby", "adult", "player", "athlete"}
    has_person = bool(set(tokens) & person_words)

    # 5. 是否包含物体
    # 通过名词密度近似：总词数 - 动词 - 功能词
    function_words = {"a", "an", "the", "is", "are", "was", "were", "in", "on",
                      "at", "to", "of", "and", "or", "with", "from", "for",
                      "by", "up", "down", "out", "into", "while", "then",
                      "their", "his", "her", "its", "this", "that"}
    content_words = [t for t in tokens if t not in function_words and t not in ALL_ACTION_VERBS and t not in person_words]
    num_content_words = len(content_words)

    # 6. 是否包含时序/顺序词
    temporal_words = {"then", "after", "before", "while", "during", "first",
                      "next", "finally", "again", "continue", "continues",
                      "still", "already", "start", "end", "begin", "finish",
                      "meanwhile", "suddenly", "slowly", "quickly", "back"}
    has_temporal_cue = bool(set(tokens) & temporal_words)

    return {
        "query_length_bucket": length_bucket,
        "num_words": num_words,
        "action_category": action_category,
        "action_complexity": action_complexity,
        "has_person_subject": has_person,
        "num_content_words": num_content_words,
        "has_temporal_cue": has_temporal_cue,
        "verbs_found": verbs_found,
    }


def extract_temporal_attributes(
    duration: float,
    gt_start: float,
    gt_end: float,
) -> Dict[str, Any]:
    """从时序区间和视频长度提取隐式属性"""
    gt_len = gt_end - gt_start
    gt_ratio = gt_len / duration if duration > 0 else 0

    # 1. 片段长度占比分桶
    if gt_ratio <= 0.1:
        coverage_bucket = "tiny(<=10%)"
    elif gt_ratio <= 0.3:
        coverage_bucket = "small(10-30%)"
    elif gt_ratio <= 0.5:
        coverage_bucket = "medium(30-50%)"
    elif gt_ratio <= 0.7:
        coverage_bucket = "large(50-70%)"
    else:
        coverage_bucket = "dominant(>70%)"

    # 2. 片段在视频中的位置
    gt_center = (gt_start + gt_end) / 2.0
    relative_pos = gt_center / duration if duration > 0 else 0.5
    if relative_pos <= 0.33:
        position_bucket = "beginning"
    elif relative_pos <= 0.66:
        position_bucket = "middle"
    else:
        position_bucket = "end"

    # 3. 视频长度分桶
    if duration <= 20:
        video_length_bucket = "very_short(<=20s)"
    elif duration <= 40:
        video_length_bucket = "short(20-40s)"
    elif duration <= 80:
        video_length_bucket = "medium(40-80s)"
    elif duration <= 160:
        video_length_bucket = "long(80-160s)"
    else:
        video_length_bucket = "very_long(>160s)"

    # 4. GT 绝对长度分桶
    if gt_len <= 3:
        gt_length_bucket = "very_short(<=3s)"
    elif gt_len <= 8:
        gt_length_bucket = "short(3-8s)"
    elif gt_len <= 20:
        gt_length_bucket = "medium(8-20s)"
    elif gt_len <= 60:
        gt_length_bucket = "long(20-60s)"
    else:
        gt_length_bucket = "very_long(>60s)"

    # 5. 是否起始于视频开头或结束于视频结尾
    starts_at_beginning = gt_start < 1.0
    ends_at_end = (duration - gt_end) < 1.0

    return {
        "gt_coverage_ratio": gt_ratio,
        "gt_coverage_bucket": coverage_bucket,
        "gt_position_bucket": position_bucket,
        "gt_relative_position": relative_pos,
        "video_length_bucket": video_length_bucket,
        "video_duration": duration,
        "gt_length": gt_len,
        "gt_length_bucket": gt_length_bucket,
        "starts_at_beginning": starts_at_beginning,
        "ends_at_end": ends_at_end,
    }


def extract_multi_query_attributes(
    annotations: List[List],
) -> Dict[str, Dict[str, Any]]:
    """
    分析同一视频内多个查询之间的关系。
    返回: {(vid, gt_start, gt_end, query): {属性字典}}
    """
    vid_to_samples = defaultdict(list)
    for anno in annotations:
        vid, duration, (gt_start, gt_end), query = anno
        vid_to_samples[vid].append((gt_start, gt_end, query))

    result = {}
    for anno in annotations:
        vid, duration, (gt_start, gt_end), query = anno
        key = (vid, gt_start, gt_end, query)

        siblings = vid_to_samples[vid]
        num_queries_in_video = len(siblings)

        # 检查是否有时序重叠的兄弟查询
        overlapping_queries = 0
        for s_start, s_end, s_query in siblings:
            if s_query == query and s_start == gt_start and s_end == gt_end:
                continue
            inter_start = max(gt_start, s_start)
            inter_end = min(gt_end, s_end)
            if inter_end > inter_start:
                overlapping_queries += 1

        # 查询密度分桶
        if num_queries_in_video == 1:
            density_bucket = "single_query"
        elif num_queries_in_video <= 3:
            density_bucket = "sparse(2-3)"
        elif num_queries_in_video <= 6:
            density_bucket = "medium(4-6)"
        else:
            density_bucket = "dense(7+)"

        result[key] = {
            "num_queries_in_video": num_queries_in_video,
            "query_density_bucket": density_bucket,
            "num_overlapping_queries": overlapping_queries,
            "has_overlapping_queries": overlapping_queries > 0,
        }

    return result


# ──────────────────── 切片评估引擎 ────────────────────

def compute_metrics(
    ious: List[float],
    thresholds: List[float],
) -> Dict[str, float]:
    """给定一组 top1 IoU 值，计算 R@1 指标"""
    n = len(ious)
    if n == 0:
        metrics = {f"R@1,IoU={thr}": 0.0 for thr in thresholds}
        metrics["mIoU"] = 0.0
        metrics["count"] = 0
        return metrics

    ious_arr = np.array(ious)
    metrics = {}
    for thr in thresholds:
        metrics[f"R@1,IoU={thr}"] = float(np.mean(ious_arr >= thr))
    metrics["mIoU"] = float(np.mean(ious_arr))
    metrics["count"] = n
    return metrics


def build_sample_index(predictions: Dict) -> Dict[str, Dict]:
    """
    从预测文件构建 (vid_name, gt_start, gt_end) -> sample 的索引。
    由于同一视频可能有多条不同 gt_window 的查询, 需要更精确的匹配。
    """
    index = {}
    for i, sample in enumerate(predictions["samples"]):
        vid = sample["vid_name"]
        gt_s, gt_e = sample["gt_window"]
        # 用序号作为备选 key 来处理完全相同 gt_window 的情况
        key = (vid, round(gt_s, 2), round(gt_e, 2), i)
        index[key] = sample
    return index


class SlicedAnalyzer:
    """切片分析器：在指定属性维度上对数据切片并计算指标"""

    def __init__(
        self,
        dataset: str,
        annotations: List[List],
        model_predictions: Dict[str, Dict],  # model_name -> prediction data
    ):
        self.dataset = dataset
        self.annotations = annotations
        self.iou_thresholds = DATASET_IOU_THRESHOLDS[dataset]
        self.model_names = list(model_predictions.keys())

        # 提取属性
        self.multi_query_attrs = extract_multi_query_attributes(annotations)

        # 构建统一的样本列表（按注释顺序），每个样本附带所有属性和各模型的预测
        self.samples = []
        # 为每个模型按顺序索引（预测文件和注释文件的样本顺序一致）
        model_samples_list = {}
        for name, pred_data in model_predictions.items():
            model_samples_list[name] = pred_data["samples"]

        for i, anno in enumerate(annotations):
            vid, duration, (gt_start, gt_end), query = anno

            text_attrs = extract_text_attributes(query)
            temporal_attrs = extract_temporal_attributes(duration, gt_start, gt_end)
            mq_key = (vid, gt_start, gt_end, query)
            mq_attrs = self.multi_query_attrs.get(mq_key, {
                "num_queries_in_video": 1,
                "query_density_bucket": "single_query",
                "num_overlapping_queries": 0,
                "has_overlapping_queries": False,
            })

            # 获取各模型的 top1 IoU
            model_ious = {}
            model_top5_ious = {}
            for name in self.model_names:
                samples_list = model_samples_list[name]
                if i < len(samples_list):
                    s = samples_list[i]
                    model_ious[name] = s["top1_iou"]
                    model_top5_ious[name] = s["top5_best_iou"]
                else:
                    model_ious[name] = 0.0
                    model_top5_ious[name] = 0.0

            self.samples.append({
                "index": i,
                "vid": vid,
                "duration": duration,
                "gt_start": gt_start,
                "gt_end": gt_end,
                "query": query,
                "text_attrs": text_attrs,
                "temporal_attrs": temporal_attrs,
                "multi_query_attrs": mq_attrs,
                "model_top1_ious": model_ious,
                "model_top5_ious": model_top5_ious,
            })

    def get_slice_dimensions(self) -> List[str]:
        """返回所有可用的切片维度"""
        return [
            # 文本属性
            "query_length_bucket",
            "action_complexity",
            "action_category",
            "has_person_subject",
            "has_temporal_cue",
            # 时序属性
            "gt_coverage_bucket",
            "gt_position_bucket",
            "video_length_bucket",
            "gt_length_bucket",
            "starts_at_beginning",
            "ends_at_end",
            # 多查询属性
            "query_density_bucket",
            "has_overlapping_queries",
        ]

    def _get_attr_value(self, sample: Dict, dim: str) -> Any:
        """从样本中获取指定属性维度的值"""
        # 文本属性
        if dim in sample["text_attrs"]:
            val = sample["text_attrs"][dim]
            # action_category 是列表，转为字符串
            if isinstance(val, list):
                return ",".join(val)
            return val
        # 时序属性
        if dim in sample["temporal_attrs"]:
            return sample["temporal_attrs"][dim]
        # 多查询属性
        if dim in sample["multi_query_attrs"]:
            return sample["multi_query_attrs"][dim]
        return None

    def analyze_by_dimension(self, dim: str) -> Dict:
        """按某一维度进行切片分析"""
        # 按属性值分组
        groups = defaultdict(list)
        for sample in self.samples:
            val = self._get_attr_value(sample, dim)
            groups[str(val)].append(sample)

        result = {
            "dimension": dim,
            "num_slices": len(groups),
            "slices": {},
        }

        for slice_name, slice_samples in sorted(groups.items()):
            slice_result = {
                "count": len(slice_samples),
                "ratio": len(slice_samples) / len(self.samples),
                "models": {},
            }

            for model_name in self.model_names:
                top1_ious = [s["model_top1_ious"][model_name] for s in slice_samples]
                top5_ious = [s["model_top5_ious"][model_name] for s in slice_samples]
                metrics = compute_metrics(top1_ious, self.iou_thresholds)
                metrics["R@5,mIoU"] = float(np.mean(top5_ious)) if top5_ious else 0.0
                for thr in self.iou_thresholds:
                    metrics[f"R@5,IoU={thr}"] = float(np.mean(np.array(top5_ious) >= thr)) if top5_ious else 0.0
                slice_result["models"][model_name] = metrics

            result["slices"][slice_name] = slice_result

        return result

    def analyze_all_dimensions(self) -> Dict[str, Dict]:
        """对所有维度进行切片分析"""
        results = {}
        for dim in self.get_slice_dimensions():
            results[dim] = self.analyze_by_dimension(dim)
        return results

    def find_model_divergences(
        self, threshold_key: Optional[str] = None, min_count: int = 30,
    ) -> List[Dict]:
        """
        找出模型之间差异最大的切片。
        返回按差异绝对值降序排列的切片列表。
        """
        if threshold_key is None:
            # 使用中间 IoU 阈值
            mid_thr = self.iou_thresholds[len(self.iou_thresholds) // 2]
            threshold_key = f"R@1,IoU={mid_thr}"

        all_results = self.analyze_all_dimensions()
        divergences = []

        for dim, dim_result in all_results.items():
            for slice_name, slice_data in dim_result["slices"].items():
                if slice_data["count"] < min_count:
                    continue

                model_values = {}
                for model_name in self.model_names:
                    model_values[model_name] = slice_data["models"][model_name].get(threshold_key, 0)

                if len(model_values) < 2:
                    continue

                values = list(model_values.values())
                max_diff = max(values) - min(values)
                best_model = max(model_values, key=model_values.get)
                worst_model = min(model_values, key=model_values.get)

                divergences.append({
                    "dimension": dim,
                    "slice": slice_name,
                    "count": slice_data["count"],
                    "metric": threshold_key,
                    "max_diff": max_diff,
                    "best_model": best_model,
                    "worst_model": worst_model,
                    "model_values": model_values,
                })

        divergences.sort(key=lambda x: x["max_diff"], reverse=True)
        return divergences

    def find_model_strengths(self, min_count: int = 30) -> Dict[str, Dict]:
        """
        为每个模型找出其擅长的和不擅长的切片。
        """
        all_results = self.analyze_all_dimensions()
        strengths = {name: {"best_slices": [], "worst_slices": []} for name in self.model_names}

        mid_thr = self.iou_thresholds[len(self.iou_thresholds) // 2]
        threshold_key = f"R@1,IoU={mid_thr}"

        for dim, dim_result in all_results.items():
            for slice_name, slice_data in dim_result["slices"].items():
                if slice_data["count"] < min_count:
                    continue

                for model_name in self.model_names:
                    val = slice_data["models"][model_name].get(threshold_key, 0)
                    entry = {
                        "dimension": dim,
                        "slice": slice_name,
                        "count": slice_data["count"],
                        threshold_key: val,
                    }
                    strengths[model_name]["best_slices"].append(entry)
                    strengths[model_name]["worst_slices"].append(entry)

        for model_name in self.model_names:
            strengths[model_name]["best_slices"].sort(
                key=lambda x: x[threshold_key], reverse=True)
            strengths[model_name]["worst_slices"].sort(
                key=lambda x: x[threshold_key])
            strengths[model_name]["best_slices"] = strengths[model_name]["best_slices"][:10]
            strengths[model_name]["worst_slices"] = strengths[model_name]["worst_slices"][:10]

        return strengths

    def cross_dimension_analysis(self, dim1: str, dim2: str) -> Dict:
        """
        二维交叉切片分析：在两个属性维度的组合上计算指标。
        用于发现更细粒度的模式。
        """
        groups = defaultdict(list)
        for sample in self.samples:
            val1 = str(self._get_attr_value(sample, dim1))
            val2 = str(self._get_attr_value(sample, dim2))
            key = f"{val1} × {val2}"
            groups[key].append(sample)

        result = {
            "dimensions": [dim1, dim2],
            "num_slices": len(groups),
            "slices": {},
        }

        for slice_name, slice_samples in sorted(groups.items()):
            if len(slice_samples) < 10:
                continue

            slice_result = {
                "count": len(slice_samples),
                "models": {},
            }

            for model_name in self.model_names:
                top1_ious = [s["model_top1_ious"][model_name] for s in slice_samples]
                metrics = compute_metrics(top1_ious, self.iou_thresholds)
                slice_result["models"][model_name] = metrics

            result["slices"][slice_name] = slice_result

        return result


# ──────────────────── 报告生成器 ────────────────────

def format_metrics_table(
    dim_result: Dict,
    model_names: List[str],
    iou_thresholds: List[float],
) -> str:
    """将切片分析结果格式化为可读表格"""
    def _format_aligned_table(headers: List[str], rows: List[List[str]]) -> List[str]:
        """生成对齐更工整的文本表格（首列左对齐，其余列右对齐）"""
        if not rows:
            widths = [len(h) for h in headers]
        else:
            widths = []
            for ci, h in enumerate(headers):
                max_cell_len = max(len(str(r[ci])) for r in rows)
                widths.append(max(len(h), max_cell_len))

        def _fmt_row(row: List[str], is_header: bool = False) -> str:
            cells = []
            for i, cell in enumerate(row):
                text = str(cell)
                if i == 0:
                    cells.append(text.ljust(widths[i]))
                else:
                    cells.append(text.rjust(widths[i]))
            return "  ".join(cells)

        header_line = _fmt_row(headers, is_header=True)
        sep_line = "-" * len(header_line)
        content_lines = [_fmt_row(r) for r in rows]
        return [header_line, sep_line] + content_lines

    lines = []
    dim_name = dim_result["dimension"]
    lines.append(f"\n{'='*80}")
    lines.append(f"  维度: {dim_name}  (共 {dim_result['num_slices']} 个切片)")
    lines.append(f"{'='*80}")

    # 构建表头
    metric_keys = [f"R@1,IoU={thr}" for thr in iou_thresholds] + ["mIoU"]
    headers = ["切片", "数量"]
    for model_name in model_names:
        short_name = model_name.replace("CPL_", "")
        for mk in metric_keys:
            short_mk = mk.replace("R@1,", "").replace("IoU=", "")
            headers.append(f"{short_name}_{short_mk}")

    table_rows = []
    for slice_name, slice_data in sorted(
        dim_result["slices"].items(),
        key=lambda x: x[1]["count"],
        reverse=True,
    ):
        row_parts = [str(slice_name), str(slice_data["count"])]
        for model_name in model_names:
            model_metrics = slice_data["models"][model_name]
            for mk in metric_keys:
                row_parts.append(f"{model_metrics.get(mk, 0) * 100:.1f}")
        table_rows.append(row_parts)

    lines.extend(_format_aligned_table(headers, table_rows))

    return "\n".join(lines)


def format_divergence_report(divergences: List[Dict], top_n: int = 20) -> str:
    """格式化模型差异报告"""
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append("  模型间差异最大的切片 (Top-{})".format(min(top_n, len(divergences))))
    lines.append(f"{'='*80}")

    for i, d in enumerate(divergences[:top_n]):
        lines.append(
            f"\n  #{i+1}  [{d['dimension']}] {d['slice']}  "
            f"(n={d['count']}, 差异={d['max_diff']*100:.1f}pp)"
        )
        lines.append(f"       最优: {d['best_model']}  |  最差: {d['worst_model']}")
        for m, v in d["model_values"].items():
            lines.append(f"         {m}: {v*100:.1f}%")

    return "\n".join(lines)


def format_strength_report(strengths: Dict[str, Dict]) -> str:
    """格式化各模型优劣势报告"""
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append("  各模型优劣势分析")
    lines.append(f"{'='*80}")

    for model_name, data in strengths.items():
        lines.append(f"\n  --- {model_name} ---")

        lines.append("  擅长 (Top-5):")
        for entry in data["best_slices"][:5]:
            metric_key = [k for k in entry if k.startswith("R@1")][0]
            lines.append(
                f"    [{entry['dimension']}] {entry['slice']}  "
                f"{metric_key}={entry[metric_key]*100:.1f}%  (n={entry['count']})"
            )

        lines.append("  不擅长 (Bottom-5):")
        for entry in data["worst_slices"][:5]:
            metric_key = [k for k in entry if k.startswith("R@1")][0]
            lines.append(
                f"    [{entry['dimension']}] {entry['slice']}  "
                f"{metric_key}={entry[metric_key]*100:.1f}%  (n={entry['count']})"
            )

    return "\n".join(lines)


def write_analysis_csv(analysis_bundle: Dict[str, Any], output_csv_path: str):
    """
    将分析结果汇总写入同一个 CSV 文件。
    CSV 采用长表格式，便于后续筛选/透视分析。
    """
    fieldnames = [
        "section",
        "dataset",
        "dimension",
        "slice",
        "count",
        "ratio",
        "model",
        "metric",
        "value",
        "extra",
    ]

    dataset = analysis_bundle["dataset"]
    model_names = analysis_bundle["model_names"]
    iou_thresholds = analysis_bundle["iou_thresholds"]
    dim_results = analysis_bundle["dimension_results"]
    divergences = analysis_bundle.get("divergences", [])
    strengths = analysis_bundle.get("strengths", {})
    global_metrics = analysis_bundle.get("global_metrics", {})

    rows = []

    # 1) 全局指标
    for model_name in model_names:
        metrics = global_metrics.get(model_name, {})
        for mk, mv in metrics.items():
            if mk == "count":
                continue
            rows.append({
                "section": "global_metrics",
                "dataset": dataset,
                "dimension": "",
                "slice": "",
                "count": metrics.get("count", 0),
                "ratio": "",
                "model": model_name,
                "metric": mk,
                "value": mv,
                "extra": "",
            })

    # 2) 各维度切片指标
    metric_keys = [f"R@1,IoU={thr}" for thr in iou_thresholds] + [
        "mIoU",
        "R@5,mIoU",
    ] + [f"R@5,IoU={thr}" for thr in iou_thresholds]

    for dim_name, dim_result in dim_results.items():
        for slice_name, slice_data in dim_result["slices"].items():
            for model_name in model_names:
                model_metrics = slice_data["models"].get(model_name, {})
                for mk in metric_keys:
                    if mk not in model_metrics:
                        continue
                    rows.append({
                        "section": "slice_metrics",
                        "dataset": dataset,
                        "dimension": dim_name,
                        "slice": slice_name,
                        "count": slice_data.get("count", 0),
                        "ratio": slice_data.get("ratio", ""),
                        "model": model_name,
                        "metric": mk,
                        "value": model_metrics.get(mk, 0.0),
                        "extra": "",
                    })

    # 3) 差异切片
    for d in divergences:
        for model_name, model_val in d.get("model_values", {}).items():
            rows.append({
                "section": "divergence",
                "dataset": dataset,
                "dimension": d.get("dimension", ""),
                "slice": d.get("slice", ""),
                "count": d.get("count", 0),
                "ratio": "",
                "model": model_name,
                "metric": d.get("metric", ""),
                "value": model_val,
                "extra": f"max_diff={d.get('max_diff', 0.0)};best={d.get('best_model', '')};worst={d.get('worst_model', '')}",
            })

    # 4) 模型优劣势
    for model_name, model_strength in strengths.items():
        for section_key in ["best_slices", "worst_slices"]:
            section_name = f"strength_{section_key}"
            for entry in model_strength.get(section_key, []):
                metric_keys_in_entry = [k for k in entry.keys() if k.startswith("R@1")]
                metric_key = metric_keys_in_entry[0] if metric_keys_in_entry else ""
                metric_val = entry.get(metric_key, 0.0) if metric_key else ""
                rows.append({
                    "section": section_name,
                    "dataset": dataset,
                    "dimension": entry.get("dimension", ""),
                    "slice": entry.get("slice", ""),
                    "count": entry.get("count", 0),
                    "ratio": "",
                    "model": model_name,
                    "metric": metric_key,
                    "value": metric_val,
                    "extra": "",
                })

    with open(output_csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV 数据已保存到: {output_csv_path}")


def write_report_json(
    analyzer: SlicedAnalyzer,
    all_dim_results: Dict[str, Dict],
    divergences: List[Dict],
    strengths: Dict[str, Dict],
    global_metrics: Dict[str, Dict[str, float]],
    output_json_path: str,
):
    """
    输出“报告版 JSON”：更偏向阅读和下游报表系统使用。
    说明：不同于 analysis_data.json（偏完整原始结构），该文件附带百分制摘要。
    """
    global_metrics_percent = {}
    for model_name, metrics in global_metrics.items():
        global_metrics_percent[model_name] = {}
        for k, v in metrics.items():
            if k == "count":
                global_metrics_percent[model_name][k] = v
            else:
                global_metrics_percent[model_name][k] = round(v * 100, 4)

    dimension_summary = {}
    for dim_name, dim_result in all_dim_results.items():
        slices_summary = []
        for slice_name, slice_data in sorted(
            dim_result["slices"].items(),
            key=lambda x: x[1]["count"],
            reverse=True,
        ):
            model_metrics_percent = {}
            for model_name in analyzer.model_names:
                model_metrics_percent[model_name] = {}
                for mk, mv in slice_data["models"][model_name].items():
                    if mk == "count":
                        model_metrics_percent[model_name][mk] = mv
                    else:
                        model_metrics_percent[model_name][mk] = round(mv * 100, 4)

            slices_summary.append({
                "slice": slice_name,
                "count": slice_data["count"],
                "ratio": round(slice_data.get("ratio", 0.0), 6),
                "models_percent": model_metrics_percent,
            })

        dimension_summary[dim_name] = {
            "dimension": dim_name,
            "num_slices": dim_result["num_slices"],
            "slices": slices_summary,
        }

    report_json = {
        "dataset": analyzer.dataset,
        "model_names": analyzer.model_names,
        "num_samples": len(analyzer.samples),
        "iou_thresholds": analyzer.iou_thresholds,
        "global_metrics_percent": global_metrics_percent,
        "dimension_summary": dimension_summary,
        "divergences": divergences,
        "strengths": strengths,
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(report_json, f, ensure_ascii=False, indent=2)

    print(f"报告 JSON 已保存到: {output_json_path}")


def generate_full_report(analyzer: SlicedAnalyzer, output_dir: str):
    """生成完整分析报告"""
    os.makedirs(output_dir, exist_ok=True)

    # 1. 各维度的切片分析
    all_dim_results = analyzer.analyze_all_dimensions()

    report_lines = []
    report_lines.append(f"数据集: {analyzer.dataset}")
    report_lines.append(f"模型: {', '.join(analyzer.model_names)}")
    report_lines.append(f"总样本数: {len(analyzer.samples)}")
    report_lines.append(f"IoU 阈值: {analyzer.iou_thresholds}")

    # 先输出全局指标
    global_metrics = {}
    report_lines.append(f"\n{'='*80}")
    report_lines.append("  全局指标")
    report_lines.append(f"{'='*80}")
    for model_name in analyzer.model_names:
        top1_ious = [s["model_top1_ious"][model_name] for s in analyzer.samples]
        metrics = compute_metrics(top1_ious, analyzer.iou_thresholds)
        global_metrics[model_name] = metrics
        report_lines.append(f"\n  {model_name}:")
        for k, v in metrics.items():
            if k == "count":
                continue
            report_lines.append(f"    {k}: {v*100:.2f}%")

    # 各维度表格
    for dim, dim_result in all_dim_results.items():
        report_lines.append(
            format_metrics_table(dim_result, analyzer.model_names, analyzer.iou_thresholds)
        )

    # 2. 模型差异分析
    divergences = analyzer.find_model_divergences()
    report_lines.append(format_divergence_report(divergences))

    # 3. 模型优劣势分析
    strengths = analyzer.find_model_strengths()
    report_lines.append(format_strength_report(strengths))

    # 4. 交叉切片分析（选几个有意义的组合）
    cross_pairs = [
        ("query_length_bucket", "gt_coverage_bucket"),
        ("action_complexity", "gt_position_bucket"),
        ("query_density_bucket", "gt_length_bucket"),
    ]

    report_lines.append(f"\n{'='*80}")
    report_lines.append("  交叉切片分析")
    report_lines.append(f"{'='*80}")

    for dim1, dim2 in cross_pairs:
        cross_result = analyzer.cross_dimension_analysis(dim1, dim2)
        if cross_result["num_slices"] > 0:
            report_lines.append(
                format_metrics_table(
                    {"dimension": f"{dim1} × {dim2}",
                     "num_slices": cross_result["num_slices"],
                     "slices": cross_result["slices"]},
                    analyzer.model_names,
                    analyzer.iou_thresholds,
                )
            )

    # 写出报告
    report_text = "\n".join(report_lines)
    prefix = analyzer.dataset
    report_path = os.path.join(output_dir, f"{prefix}_analysis_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"报告已保存到: {report_path}")

    # 保存 JSON 格式结果（方便后续可视化）
    json_result = {
        "dataset": analyzer.dataset,
        "model_names": analyzer.model_names,
        "num_samples": len(analyzer.samples),
        "iou_thresholds": analyzer.iou_thresholds,
        "global_metrics": global_metrics,
        "dimension_results": {},
        "divergences": divergences[:30],
        "strengths": strengths,
    }
    for dim, dim_result in all_dim_results.items():
        json_result["dimension_results"][dim] = dim_result

    json_path = os.path.join(output_dir, f"{prefix}_analysis_data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_result, f, ensure_ascii=False, indent=2)
    print(f"JSON 数据已保存到: {json_path}")

    # 保存 CSV（同一文件汇总全部分析内容）
    csv_path = os.path.join(output_dir, f"{prefix}_analysis_report.csv")
    write_analysis_csv(json_result, csv_path)

    # 保存报告版 JSON（可读、百分制）
    report_json_path = os.path.join(output_dir, f"{prefix}_analysis_report.json")
    write_report_json(
        analyzer=analyzer,
        all_dim_results=all_dim_results,
        divergences=divergences,
        strengths=strengths,
        global_metrics=global_metrics,
        output_json_path=report_json_path,
    )

    # 输出关键发现到终端
    print("\n" + report_text)

    return json_result


# ──────────────────── CLI ────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="细粒度切片分析：对比模型在不同属性子集上的表现")
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=["charades", "activitynet"],
        help="数据集名称")
    parser.add_argument(
        "--pred-files", type=str, nargs="+", required=True,
        help="模型预测 JSON 文件路径（可多个）")
    parser.add_argument(
        "--output-dir", type=str, default="sliced_results",
        help="输出目录")
    parser.add_argument(
        "--min-slice-count", type=int, default=30,
        help="切片最小样本数（低于此数的切片在差异分析中忽略）")
    return parser.parse_args()


def main():
    args = parse_args()

    # 加载注释
    print(f"加载 {args.dataset} 注释文件...")
    annotations = load_annotations(args.dataset)
    print(f"共 {len(annotations)} 个样本")

    # 加载各模型预测
    model_predictions = {}
    for pred_file in args.pred_files:
        # 自动处理相对路径
        if not os.path.isabs(pred_file):
            pred_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), pred_file)

        print(f"加载预测文件: {pred_file}")
        pred_data = load_predictions(pred_file)
        model_name = pred_data["meta"]["model_name"]
        model_predictions[model_name] = pred_data
        print(f"  模型: {model_name}, 样本数: {len(pred_data['samples'])}")

    # 创建分析器
    analyzer = SlicedAnalyzer(args.dataset, annotations, model_predictions)

    # 生成报告
    generate_full_report(analyzer, args.output_dir)


if __name__ == "__main__":
    main()