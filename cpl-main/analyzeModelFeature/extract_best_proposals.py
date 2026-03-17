import json
import argparse
from collections import defaultdict


def _iou_1d(gt_start, gt_end, pred_start, pred_end):
    """计算 1D IoU"""
    inter_start = max(gt_start, pred_start)
    inter_end = min(gt_end, pred_end)
    inter_len = max(0.0, inter_end - inter_start)
    union_len = (pred_end - pred_start) + (gt_end - gt_start) - inter_len
    return inter_len / union_len if union_len > 0 else 0.0


def extract_best_proposals(input_json_path, output_json_path=None):
    """
    从 AncPropInfo.json 中提取每个视频样本 IoU 最高的 proposal
    并计算 R@1 指标
    """
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = data['samples']
    
    # 提取最高 IoU 的 proposal
    best_proposals = []
    iou_thresholds = [0.1, 0.3, 0.5]
    
    # 初始化指标统计
    metrics = {
        'r@1_iou0.1': 0,
        'r@1_iou0.3': 0,
        'r@1_iou0.5': 0,
    }
    miou_sum = 0.0
    
    for sample in samples:
        proposals = sample['proposals']
        gt_window = sample['gt_window']
        
        # 找到 IoU 最高的 proposal
        best_proposal = max(proposals, key=lambda p: p['iou'])
        best_iou = best_proposal['iou']
        
        # 计算 R@1 指标
        for thr in iou_thresholds:
            if best_iou >= thr:
                metrics[f'r@1_iou{thr}'] += 1
        
        miou_sum += best_iou
        
        best_proposals.append({
            'vid_name': sample['vid_name'],
            'duration': sample['duration'],
            'gt_window': gt_window,
            'best_proposal': {
                'pred_window': best_proposal['pred_window'],
                'iou': best_iou,
                'rank': best_proposal['rank'],
                'score': best_proposal['score'],
            }
        })
    
    # 计算平均指标
    num_samples = len(samples)
    r1_miou = miou_sum / num_samples if num_samples > 0 else 0.0
    
    for key in metrics:
        metrics[key] = metrics[key] / num_samples if num_samples > 0 else 0.0
    
    metrics['r@1_miou'] = r1_miou
    
    # 构建结果
    result = {
        'meta': {
            'source': input_json_path,
            'num_samples': num_samples,
            'description': 'Best proposals extracted from AncPropInfo.json (highest IoU per sample)',
        },
        'metrics': metrics,
        'best_proposals': best_proposals,
    }
    
    # 保存结果
    if output_json_path:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {output_json_path}")
    
    # 打印性能指标
    print("\n" + "="*60)
    print("Performance Metrics (Best Proposals)")
    print("="*60)
    print(f"Number of samples: {num_samples}")
    print(f"R@1 IoU=0.1: {metrics['r@1_iou0.1']:.4f}")
    print(f"R@1 IoU=0.3: {metrics['r@1_iou0.3']:.4f}")
    print(f"R@1 IoU=0.5: {metrics['r@1_iou0.5']:.4f}")
    print(f"R@1 mIoU:    {metrics['r@1_miou']:.4f}")
    print("="*60 + "\n")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Extract best proposals from AncPropInfo.json and compute R@1 metrics"
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Path to AncPropInfo.json')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save results (optional)')
    
    args = parser.parse_args()
    
    extract_best_proposals(args.input, args.output)


if __name__ == '__main__':
    main()
