# noinspection PyUnresolvedReferences,PyPackageRequirements
# 必须在导入项目模块之前把 cpl-main 根目录加入 sys.path。你要是再敢动我的代码我他妈给你删了
import os
import sys
import argparse
import json
import random
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from runners import MainRunner
from utils import load_json


def _to_python_scalar(x):
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return x.item()
        return x.detach().cpu().numpy().tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def _to_python_list(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def _iou_1d(gt_start, gt_end, pred_start, pred_end):
    inter_start = max(gt_start, pred_start)
    inter_end = min(gt_end, pred_end)
    inter_len = max(0.0, inter_end - inter_start)
    union_len = (pred_end - pred_start) + (gt_end - gt_start) - inter_len
    return inter_len / union_len if union_len > 0 else 0.0


def analyze_model_predictions(
    model,
    dataloader,
    device,
    output_file="proposal_analysis.json",
    split_name="eval",
    iou_thresholds=(0.3, 0.5, 0.7),
    epoch_for_inference=100,
    dataset_name="ActivityNet",
    model_name="UnknownModel",
):
    """
    在验证/测试集上运行一次推理，记录每个样本的全部 num_props 提议与 GT 区间。
    同时给出 Rank@1 与 Rank@5 的诊断统计，用于分析 Rank1 大幅下降但 Rank5 提升的现象。

    参数:
        model: 训练好的模型
        dataloader: 验证/测试 dataloader
        device: 设备, 如 "cuda" 或 torch.device("cuda")
        output_file: 输出 JSON 文件路径
        split_name: 数据划分名, 如 "val" / "test"
        iou_thresholds: 评估 IoU 阈值
        epoch_for_inference: 透传给 model.forward 的 epoch 参数(用于兼容现有训练逻辑)
        dataset_name: 数据集名称
    """
    from models.loss import cal_nll_loss

    model.eval()
    analysis_results = []

    # 统计指标（重点对比 Rank1 与 Rank5）
    total_samples = 0
    rank1_shorter_than_gt = 0
    rank1_longer_than_gt = 0
    rank1_center_outside_gt = 0
    rank1_not_best_iou = 0
    rank5_has_best_iou = 0
    best_iou_rank_counts = {i: 0 for i in range(1, 9)} # 记录最好IoU出现在Rank 1~8的次数
    rank_norm_lengths = {i: [] for i in range(1, 9)} # 记录Rank 1~8的提议的归一化长度

    metrics = {
        f"r@1_iou{thr}": 0 for thr in iou_thresholds
    }
    metrics.update({
        f"r@5_iou{thr}": 0 for thr in iou_thresholds
    })

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Analyzing Predictions")):
            net_input = batch_data['net_input']
            frames_feat = net_input['frames_feat'].to(device)
            frames_len = net_input['frames_len'].to(device)
            words_id = net_input['words_id'].to(device)
            words_feat = net_input['words_feat'].to(device)
            words_len = net_input['words_len'].to(device)
            weights = net_input['weights'].to(device)

            # [vid, duration, [gt_start, gt_end], sentence]
            raw_items = batch_data['raw']

            bsz = frames_feat.shape[0]

            outputs = model(
                frames_feat=frames_feat,
                frames_len=frames_len,
                words_id=words_id,
                words_feat=words_feat,
                words_len=words_len,
                weights=weights,
                epoch=epoch_for_inference,
            )

            words_logit = outputs['words_logit']
            center = outputs['center']
            width = outputs['width']
            words_mask = outputs['words_mask']

            # 计算每个 proposal 的 nll_loss（与 main_runner.py 中的 eval 方法一致）
            words_mask_expanded = words_mask.unsqueeze(1) \
                .expand(bsz, model.num_props, -1).contiguous().view(bsz * model.num_props, -1)
            words_id_expanded = words_id.unsqueeze(1) \
                .expand(bsz, model.num_props, -1).contiguous().view(bsz * model.num_props, -1)

            nll_loss, acc = cal_nll_loss(
                words_logit, words_id_expanded, words_mask_expanded)
            nll_loss = nll_loss.view(bsz, model.num_props)

            centers = center.view(bsz, model.num_props)
            widths = width.view(bsz, model.num_props)

            for i in range(bsz):
                nll_losses_i = nll_loss[i]
                centers_i = centers[i]
                widths_i = widths[i]

                vid_name = raw_items[i][0]
                duration = float(_to_python_scalar(raw_items[i][1]))
                gt_window_i = _to_python_list(raw_items[i][2])
                gt_start, gt_end = float(gt_window_i[0]), float(gt_window_i[1])
                gt_len = gt_end - gt_start

                # 计算标准化交叉熵损失：nll_loss * width（标准化长度）
                normalized_nll_loss = nll_losses_i * widths_i
                
                # 全部 num_props 提议（按标准化 nll_loss 从小到大排序）
                sorted_normalized_nll_losses, sorted_indices = torch.sort(
                    normalized_nll_loss, descending=False)
                sample_props = []

                for rank, idx_tensor in enumerate(sorted_indices):
                    idx = int(idx_tensor.item())
                    c_norm = float(centers_i[idx].item())
                    w_norm = float(widths_i[idx].item())
                    nll_loss_val = float(nll_losses_i[idx].item())
                    normalized_nll_loss_val = float(normalized_nll_loss[idx].item())

                    c = c_norm * duration
                    w = w_norm * duration
                    pred_start = max(0.0, c - w / 2)
                    pred_end = min(duration, c + w / 2)

                    iou = _iou_1d(gt_start, gt_end, pred_start, pred_end)

                    pred_length = pred_end - pred_start
                    pred_length_norm = pred_length / duration if duration > 0 else 0.0

                    sample_props.append({
                        "proposal_index": idx,
                        "rank": rank + 1,
                        "nll_loss": nll_loss_val,
                        "normalized_nll_loss": normalized_nll_loss_val,
                        "center_norm": c_norm,
                        "width_norm": w_norm,
                        "pred_window": [pred_start, pred_end],
                        "pred_length": pred_length,
                        "pred_length_norm": pred_length_norm,
                        "iou": iou
                    })

                    # 记录Rank 1~8的标准化长度
                    if rank + 1 <= 8:
                        rank_norm_lengths[rank + 1].append(pred_length_norm)

                top1 = sample_props[0]
                top5 = sample_props[:5]
                top1_iou = top1['iou']
                top5_best_iou = max([p['iou']
                                    for p in top5]) if len(top5) > 0 else 0.0

                for thr in iou_thresholds:
                    if top1_iou >= thr:
                        metrics[f"r@1_iou{thr}"] += 1
                    if top5_best_iou >= thr:
                        metrics[f"r@5_iou{thr}"] += 1

                rank1_pred = top1['pred_window']
                rank1_len = rank1_pred[1] - rank1_pred[0]
                if rank1_len < gt_len * 0.5:
                    rank1_shorter_than_gt += 1
                elif rank1_len > gt_len * 1.5:
                    rank1_longer_than_gt += 1

                rank1_center = (rank1_pred[0] + rank1_pred[1]) / 2.0
                if not (gt_start <= rank1_center <= gt_end):
                    rank1_center_outside_gt += 1

                # 寻找全局(或所有proposal中)最佳的IoU及对应的Rank
                best_iou = -1.0
                best_rank = -1
                for p in sample_props:
                    if p['iou'] > best_iou:
                        best_iou = p['iou']
                        best_rank = p['rank']
                
                # Rank 1 不是最佳IoU的比例
                if best_rank != 1:
                    rank1_not_best_iou += 1
                
                # 最佳IoU落在前5的比例
                if 1 <= best_rank <= 5:
                    rank5_has_best_iou += 1
                
                # 统计最高IoU落在哪一个Rank(1~8)
                if 1 <= best_rank <= 8:
                    best_iou_rank_counts[best_rank] += 1

                total_samples += 1

                analysis_results.append({
                    "vid_name": str(_to_python_scalar(vid_name)),
                    "split": split_name,
                    "batch_idx": int(batch_idx),
                    "sample_idx_in_batch": int(i),
                    "duration": duration,
                    "gt_window": [gt_start, gt_end],
                    "num_props": int(model.num_props),
                    "top1_iou": top1_iou,
                    "top5_best_iou": top5_best_iou,
                    "nll_loss_margin_1_vs_2": (
                        sample_props[0]['nll_loss'] -
                        sample_props[1]['nll_loss']
                        if len(sample_props) > 1 else None
                    ),
                    "proposals": sample_props,
                })

    # 归一化统计
    if total_samples > 0:
        summary_metrics = {
            k: v / total_samples for k, v in metrics.items()
        }
        avg_rank_norm_lengths = {
            f"rank{k}": float(np.mean(lengths)) if len(lengths) > 0 else 0.0
            for k, lengths in rank_norm_lengths.items()
        }
        summary = {
            "num_samples": total_samples,
            "rank1_shorter_than_gt_ratio": rank1_shorter_than_gt / total_samples,
            "rank1_longer_than_gt_ratio": rank1_longer_than_gt / total_samples,
            "rank1_center_outside_gt_ratio": rank1_center_outside_gt / total_samples,
            "rank1_not_best_iou_ratio": rank1_not_best_iou / total_samples,
            "rank5_has_best_iou_ratio": rank5_has_best_iou / total_samples,
            "best_iou_in_rank_distribution": {f"rank{k}": v / total_samples for k, v in best_iou_rank_counts.items()},
            "avg_rank_norm_lengths": avg_rank_norm_lengths,
            "metrics": summary_metrics,
        }
    else:
        summary = {
            "num_samples": 0,
            "rank1_shorter_than_gt_ratio": 0.0,
            "rank1_longer_than_gt_ratio": 0.0,
            "rank1_center_outside_gt_ratio": 0.0,
            "rank1_not_best_iou_ratio": 0.0,
            "rank5_has_best_iou_ratio": 0.0,
            "best_iou_in_rank_distribution": {f"rank{k}": 0.0 for k in range(1, 9)},
            "avg_rank_norm_lengths": {f"rank{k}": 0.0 for k in range(1, 9)},
            "metrics": {k: 0.0 for k in metrics.keys()},
        }

    result_package = {
        "meta": {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "created_at": datetime.now().isoformat(),
            "split": split_name,
            "num_props": int(getattr(model, 'num_props', -1)),
            "output_file": os.path.abspath(output_file),
            "description": "Each sample contains all proposals from one forward inference and IoU against GT.",
        },
        "summary": summary,
        "samples": analysis_results,
    }

    output_dir = os.path.dirname(os.path.abspath(output_file))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_package, f, ensure_ascii=False, indent=2)

    print(f"Analysis complete. Results saved to {output_file}")
    print(f"Num samples: {summary['num_samples']}")
    print(
        f"Rank1 shorter-than-GT ratio: {summary['rank1_shorter_than_gt_ratio']:.4f}")
    print(
        f"Rank1 longer-than-GT ratio: {summary['rank1_longer_than_gt_ratio']:.4f}")
    print(
        f"Rank1 center-outside-GT ratio: {summary['rank1_center_outside_gt_ratio']:.4f}")
    print(
        f"Rank1 not best IoU ratio: {summary['rank1_not_best_iou_ratio']:.4f}")
    print(
        f"Rank5 has best IoU ratio: {summary['rank5_has_best_iou_ratio']:.4f}")
    print("Best IoU Rank Distribution:")
    for k, v in summary['best_iou_in_rank_distribution'].items():
        print(f"  {k}: {v:.4f}")
    print("Average Standardized Proposal Length per Rank (Rank 1~8):")
    for k, v in summary['avg_rank_norm_lengths'].items():
        print(f"  {k}: {v:.4f}")

    for thr in iou_thresholds:
        print(f"R@1 IoU={thr}: {summary['metrics'][f'r@1_iou{thr}']:.4f}")
        print(f"R@5 IoU={thr}: {summary['metrics'][f'r@5_iou{thr}']:.4f}")

    return result_package


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run proposal-level inference analysis for CPL/CPL-MoE.")
    parser.add_argument('--config-path', type=str,
                        required=True, help='Path to config json')
    parser.add_argument('--model-path', type=str,
                        required=True, help='Path to checkpoint .pt')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='Dataset split for analysis')
    parser.add_argument('--output-file', type=str, default='analysis/proposal_analysis.json',
                        help='Output json file path')
    parser.add_argument('--seed', type=int, default=8, help='Random seed')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size from config')
    parser.add_argument('--epoch-for-inference', type=int, default=100,
                        help='Epoch value passed into model forward')
    parser.add_argument('--iou-thresholds', type=float, nargs='+', default=[0.3, 0.5, 0.7],
                        help='IoU thresholds used in summary metrics')
    return parser.parse_args()


def build_runner_from_args(args):
    config = load_json(args.config_path)

    # 将配置中的相对路径转换为绝对路径（相对于项目根目录）
    dataset_config = config['dataset']

    # 处理数据和词汇表路径
    for key in ['train_data', 'test_data', 'val_data', 'vocab_path', 'feature_path']:
        if key in dataset_config:
            dataset_config[key] = _resolve_path(
                dataset_config[key], PROJECT_ROOT)

    if args.batch_size is not None:
        config['train']['batch_size'] = int(args.batch_size)

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.cuda.manual_seed(seed + 4)
    torch.cuda.manual_seed_all(seed + 4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    runner = MainRunner(config)
    runner._load_model(args.model_path)
    return runner


def select_dataloader(runner, split):
    if split == 'train':
        return runner.train_loader
    if split == 'val':
        if runner.val_loader is None:
            raise ValueError(
                'val split requested but val_loader is None. Check config[dataset][val_data].')
        return runner.val_loader
    return runner.test_loader


def _resolve_path(raw, base_dir):
    """若 raw 是相对路径，则拼接到 base_dir 下；若是绝对路径则原样返回。"""
    if os.path.isabs(raw):
        return raw
    return os.path.join(base_dir, raw)


def _safe_name(name: str):
    return ''.join(ch if (ch.isalnum() or ch in ('-', '_')) else '_' for ch in str(name))


def main():
    args = parse_args()

    # 相对路径自动补全：config/ checkpoints/
    args.config_path = _resolve_path(args.config_path,
                                     os.path.join(PROJECT_ROOT, 'config'))
    args.model_path = _resolve_path(args.model_path,
                                    os.path.join(PROJECT_ROOT, 'checkpoints'))

    if not os.path.exists(args.config_path):
        raise FileNotFoundError(f'Config file not found: {args.config_path}')
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f'Model checkpoint not found: {args.model_path}')

    # 先读取配置，获取数据集名和模型名
    config = load_json(args.config_path)
    dataset_name = config['dataset'].get('dataset', 'ActivityNet')
    model_name = config.get('model', {}).get('name', 'UnknownModel')

    # 输出到当前工作目录，按“数据集_模型名”命名
    args.output_file = f"{_safe_name(dataset_name)}_{_safe_name(model_name)}.json"

    runner = build_runner_from_args(args)

    # 根据数据集类型确定 IoU 阈值
    if dataset_name == 'CharadesSTA':
        iou_thresholds = (0.3, 0.5, 0.7)
    else:  # ActivityNet
        iou_thresholds = (0.1, 0.3, 0.5)

    dataloader = select_dataloader(runner, args.split)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    runner.model = runner.model.to(device)

    analyze_model_predictions(
        model=runner.model,
        dataloader=dataloader,
        device=device,
        output_file=args.output_file,
        split_name=args.split,
        iou_thresholds=iou_thresholds,
        epoch_for_inference=args.epoch_for_inference,
        dataset_name=dataset_name,
        model_name=model_name,
    )

# 调用示例 (你需要根据你的实际启动脚本调整传入的变量):
# analyze_model_predictions(model, val_loader, device='cuda', output_file='analysis/val_proposals.json', split_name='val')
# analyze_model_predictions(model, test_loader, device='cuda', output_file='analysis/test_proposals.json', split_name='test')


if __name__ == '__main__':
    main()

'''
python analyzeModelProposal.py --config-path activitynet/main_fusion.json --model-path activitynet_fusion/cpl_fusion/model-best.pt
'''
