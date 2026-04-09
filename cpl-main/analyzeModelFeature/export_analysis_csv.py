"""
将切片分析结果 JSON 汇总导出到单个 CSV 文件。

用法:
    python export_analysis_csv.py \
        --input sliced_results/activitynet/activitynet_analysis_data.json

    python export_analysis_csv.py \
        --input sliced_results/charades/charades_analysis_data.json \
        --output sliced_results/charades/charades_analysis_report.csv
"""

import os
import json
import argparse

from sliced_analysis import write_analysis_csv


def parse_args():
    parser = argparse.ArgumentParser(description="将 analysis_data.json 导出为单一 CSV")
    parser.add_argument(
        "--input", type=str, required=True,
        help="输入 JSON 文件路径（建议 *_analysis_data.json）")
    parser.add_argument(
        "--output", type=str, default=None,
        help="输出 CSV 路径；默认与输入同目录，命名为 <dataset>_analysis_report.csv")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        analysis_bundle = json.load(f)

    dataset = analysis_bundle.get("dataset", "dataset")

    if args.output:
        output_csv = args.output
    else:
        input_dir = os.path.dirname(os.path.abspath(args.input))
        output_csv = os.path.join(input_dir, f"{dataset}_analysis_report.csv")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    write_analysis_csv(analysis_bundle, output_csv)


if __name__ == "__main__":
    main()
