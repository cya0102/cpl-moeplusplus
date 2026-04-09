"""
可视化模块 —— 基于 sliced_analysis.py 输出的 analysis_data.json 生成对比图表。

用法:
    python visualize_sliced.py \
        --input sliced_results/charades/analysis_data.json \
        --output-dir sliced_results/charades/figures

    # 也可以直接从预测文件运行（会先调用分析再可视化）:
    python visualize_sliced.py \
        --dataset charades \
        --pred-files CharadesSTA_CPL_Fusion.json CharadesSTA_CPL_DMVS.json \
        --output-dir sliced_results/charades/figures
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 中文显示支持
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 配色
COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
]


def load_analysis_data(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ──────────────── 柱状对比图 ────────────────

def plot_bar_comparison(
    dim_result: Dict,
    model_names: List[str],
    metric_key: str,
    title: str,
    output_path: str,
    figsize=(12, 5),
    min_count: int = 20,
):
    """为某一维度的各切片画分组柱状图"""
    slices = dim_result["slices"]

    # 过滤太小的切片
    filtered = {k: v for k, v in slices.items() if v["count"] >= min_count}
    if not filtered:
        return

    slice_names = sorted(filtered.keys())
    x = np.arange(len(slice_names))
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=figsize)

    for i, model_name in enumerate(model_names):
        values = []
        for sn in slice_names:
            v = filtered[sn]["models"][model_name].get(metric_key, 0)
            values.append(v * 100)
        short_name = model_name.replace("CPL_", "")
        bars = ax.bar(
            x + i * width - (len(model_names) - 1) * width / 2,
            values, width,
            label=short_name,
            color=COLORS[i % len(COLORS)],
            edgecolor="white", linewidth=0.5,
        )
        # 在柱顶标注数值
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=7)

    # 在 x 标签下方标注样本数
    labels = []
    for sn in slice_names:
        count = filtered[sn]["count"]
        labels.append(f"{sn}\n(n={count})")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel(f"{metric_key} (%)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────── 雷达图 ────────────────

def plot_radar_chart(
    dim_result: Dict,
    model_names: List[str],
    metric_key: str,
    title: str,
    output_path: str,
    min_count: int = 20,
):
    """雷达图展示各模型在各切片上的表现"""
    slices = dim_result["slices"]
    filtered = {k: v for k, v in slices.items() if v["count"] >= min_count}
    if len(filtered) < 3:
        return

    categories = sorted(filtered.keys())
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i, model_name in enumerate(model_names):
        values = []
        for cat in categories:
            v = filtered[cat]["models"][model_name].get(metric_key, 0)
            values.append(v * 100)
        values += values[:1]

        short_name = model_name.replace("CPL_", "")
        ax.plot(angles, values, 'o-', linewidth=2, label=short_name,
                color=COLORS[i % len(COLORS)], markersize=4)
        ax.fill(angles, values, alpha=0.1, color=COLORS[i % len(COLORS)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────── 差异热力图 ────────────────

def plot_divergence_heatmap(
    analysis_data: Dict,
    metric_key: str,
    output_path: str,
    min_count: int = 30,
    figsize=(14, 10),
):
    """
    热力图展示各维度×切片上模型间的差异程度。
    颜色深浅 = 模型间最大差异。
    """
    model_names = analysis_data["model_names"]
    if len(model_names) < 2:
        return

    dim_results = analysis_data["dimension_results"]

    rows = []  # (dim, slice_name, diff, values_dict)
    for dim_name, dim_result in dim_results.items():
        for slice_name, slice_data in dim_result["slices"].items():
            if slice_data["count"] < min_count:
                continue
            values = {}
            for m in model_names:
                values[m] = slice_data["models"][m].get(metric_key, 0) * 100
            vals = list(values.values())
            diff = max(vals) - min(vals)
            rows.append((dim_name, slice_name, diff, values, slice_data["count"]))

    rows.sort(key=lambda x: x[2], reverse=True)
    rows = rows[:25]  # Top-25

    if not rows:
        return

    fig, ax = plt.subplots(figsize=figsize)

    y_labels = []
    data_matrix = []
    for dim_name, slice_name, diff, values, count in rows:
        y_labels.append(f"[{dim_name}] {slice_name} (n={count})")
        row_data = [values[m] for m in model_names]
        data_matrix.append(row_data)

    data_matrix = np.array(data_matrix)

    # 标准化每行以突出差异
    im = ax.imshow(data_matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels([m.replace("CPL_", "") for m in model_names], fontsize=10)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=8)

    # 在每个单元格标注数值
    for i in range(len(y_labels)):
        for j in range(len(model_names)):
            ax.text(j, i, f"{data_matrix[i, j]:.1f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if data_matrix[i, j] > np.mean(data_matrix) else "black")

    ax.set_title(f"模型差异热力图 ({metric_key})\n差异最大的 Top-{len(rows)} 切片",
                 fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.6, label=f"{metric_key} (%)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────── 散点图：IoU 分布 ────────────────

def plot_iou_scatter(
    analysis_data: Dict,
    metric_key: str,
    output_path: str,
    min_count: int = 20,
):
    """
    散点图：X = 模型A的指标, Y = 模型B的指标。
    每个点是一个切片。如果有 >2 个模型，只对比前两个。
    """
    model_names = analysis_data["model_names"]
    if len(model_names) < 2:
        return

    m1, m2 = model_names[0], model_names[1]
    dim_results = analysis_data["dimension_results"]

    xs, ys, labels, sizes = [], [], [], []
    for dim_name, dim_result in dim_results.items():
        for slice_name, slice_data in dim_result["slices"].items():
            if slice_data["count"] < min_count:
                continue
            v1 = slice_data["models"][m1].get(metric_key, 0) * 100
            v2 = slice_data["models"][m2].get(metric_key, 0) * 100
            xs.append(v1)
            ys.append(v2)
            labels.append(f"{dim_name}: {slice_name}")
            sizes.append(max(20, min(200, slice_data["count"] / 2)))

    if not xs:
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    xs, ys = np.array(xs), np.array(ys)
    ax.scatter(xs, ys, s=sizes, alpha=0.6, edgecolors="black", linewidths=0.5, c=COLORS[0])

    # 对角线
    lim_min = min(min(xs), min(ys)) - 5
    lim_max = max(max(xs), max(ys)) + 5
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", alpha=0.3, linewidth=1)

    # 标注离对角线最远的点
    diffs = np.abs(xs - ys)
    top_indices = diffs.argsort()[-5:]
    for idx in top_indices:
        ax.annotate(labels[idx], (xs[idx], ys[idx]), fontsize=6,
                    xytext=(5, 5), textcoords="offset points",
                    arrowprops=dict(arrowstyle="-", alpha=0.3))

    short_m1 = m1.replace("CPL_", "")
    short_m2 = m2.replace("CPL_", "")
    ax.set_xlabel(f"{short_m1} - {metric_key} (%)", fontsize=11)
    ax.set_ylabel(f"{short_m2} - {metric_key} (%)", fontsize=11)
    ax.set_title(f"{short_m1} vs {short_m2} 按切片对比\n(点大小 = 样本数)", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────── 瀑布图：模型增益分析 ────────────────

def plot_waterfall(
    analysis_data: Dict,
    metric_key: str,
    output_path: str,
    min_count: int = 30,
):
    """
    瀑布图：展示模型 B 相对于 模型 A 在各切片上的增益/退化。
    仅当有 >= 2 个模型时生效。
    """
    model_names = analysis_data["model_names"]
    if len(model_names) < 2:
        return

    m1, m2 = model_names[0], model_names[1]
    dim_results = analysis_data["dimension_results"]

    entries = []
    for dim_name, dim_result in dim_results.items():
        for slice_name, slice_data in dim_result["slices"].items():
            if slice_data["count"] < min_count:
                continue
            v1 = slice_data["models"][m1].get(metric_key, 0) * 100
            v2 = slice_data["models"][m2].get(metric_key, 0) * 100
            diff = v2 - v1
            entries.append({
                "label": f"[{dim_name}] {slice_name}",
                "diff": diff,
                "count": slice_data["count"],
                "v1": v1,
                "v2": v2,
            })

    entries.sort(key=lambda x: x["diff"])

    # 取两端各10个
    if len(entries) > 20:
        entries = entries[:10] + entries[-10:]

    if not entries:
        return

    fig, ax = plt.subplots(figsize=(10, max(6, len(entries) * 0.35)))

    y_pos = range(len(entries))
    diffs = [e["diff"] for e in entries]
    colors = ["#C44E52" if d < 0 else "#55A868" for d in diffs]
    labels = [e["label"] for e in entries]

    ax.barh(y_pos, diffs, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)

    for i, e in enumerate(entries):
        x = e["diff"]
        offset = 0.5 if x >= 0 else -0.5
        ax.text(x + offset, i, f"{x:+.1f}", va="center", fontsize=7,
                ha="left" if x >= 0 else "right")

    short_m1 = m1.replace("CPL_", "")
    short_m2 = m2.replace("CPL_", "")
    ax.set_xlabel(f"{metric_key} 差异 (pp): {short_m2} - {short_m1}", fontsize=10)
    ax.set_title(f"{short_m2} 相对 {short_m1} 的增益/退化", fontsize=12, fontweight="bold")
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────── 主流程 ────────────────

def generate_all_figures(analysis_data: Dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    model_names = analysis_data["model_names"]
    iou_thresholds = analysis_data["iou_thresholds"]
    dim_results = analysis_data["dimension_results"]

    # 选择主要指标
    mid_thr = iou_thresholds[len(iou_thresholds) // 2]
    primary_metric = f"R@1,IoU={mid_thr}"

    # 关键维度（用于柱状图和雷达图）
    key_dims = [
        "query_length_bucket",
        "action_complexity",
        "gt_coverage_bucket",
        "gt_position_bucket",
        "video_length_bucket",
        "gt_length_bucket",
        "query_density_bucket",
    ]

    print(f"生成图表到 {output_dir} ...")

    # 1. 每个关键维度的柱状图
    for dim in key_dims:
        if dim not in dim_results:
            continue
        plot_bar_comparison(
            dim_results[dim], model_names, primary_metric,
            title=f"{dim} — {primary_metric}",
            output_path=os.path.join(output_dir, f"bar_{dim}.png"),
        )
        # mIoU 的柱状图
        plot_bar_comparison(
            dim_results[dim], model_names, "mIoU",
            title=f"{dim} — mIoU",
            output_path=os.path.join(output_dir, f"bar_{dim}_miou.png"),
        )

    # 2. 每个关键维度的雷达图
    for dim in key_dims:
        if dim not in dim_results:
            continue
        plot_radar_chart(
            dim_results[dim], model_names, primary_metric,
            title=f"{dim} — {primary_metric}",
            output_path=os.path.join(output_dir, f"radar_{dim}.png"),
        )

    # 3. 差异热力图
    plot_divergence_heatmap(
        analysis_data, primary_metric,
        output_path=os.path.join(output_dir, "heatmap_divergence.png"),
    )

    # 4. 散点对比图
    plot_iou_scatter(
        analysis_data, primary_metric,
        output_path=os.path.join(output_dir, "scatter_comparison.png"),
    )

    # 5. 瀑布图
    plot_waterfall(
        analysis_data, primary_metric,
        output_path=os.path.join(output_dir, "waterfall_gain.png"),
    )

    # 6. 针对所有 IoU 阈值分别画一张汇总柱状图
    for thr in iou_thresholds:
        mk = f"R@1,IoU={thr}"
        plot_bar_comparison(
            dim_results.get("gt_coverage_bucket", {"slices": {}}),
            model_names, mk,
            title=f"GT Coverage — {mk}",
            output_path=os.path.join(output_dir, f"bar_coverage_{mk.replace(',', '_').replace('=', '')}.png"),
        )

    print(f"所有图表已保存到 {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="生成切片分析的可视化图表")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=str, help="分析结果 JSON (analysis_data.json)")
    group.add_argument("--dataset", type=str, choices=["charades", "activitynet"],
                       help="数据集（需同时提供 --pred-files）")
    parser.add_argument("--pred-files", type=str, nargs="+",
                        help="预测文件（与 --dataset 配合使用）")
    parser.add_argument("--output-dir", type=str, default="figures",
                        help="图表输出目录")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.input:
        analysis_data = load_analysis_data(args.input)
    else:
        if not args.pred_files:
            print("使用 --dataset 时必须同时提供 --pred-files")
            sys.exit(1)
        # 先运行分析
        from sliced_analysis import (
            load_annotations, load_predictions, SlicedAnalyzer, generate_full_report
        )
        annotations = load_annotations(args.dataset)
        model_predictions = {}
        for pf in args.pred_files:
            if not os.path.isabs(pf):
                pf = os.path.join(os.path.dirname(os.path.abspath(__file__)), pf)
            pred_data = load_predictions(pf)
            model_predictions[pred_data["meta"]["model_name"]] = pred_data

        analyzer = SlicedAnalyzer(args.dataset, annotations, model_predictions)
        analysis_data = generate_full_report(analyzer, args.output_dir)

    generate_all_figures(analysis_data, args.output_dir)


if __name__ == "__main__":
    main()
