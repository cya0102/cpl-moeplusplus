"""
基于 AncPropInfo1.json 中的数据，按照 weighted_nll_loss 重新排序提议，
计算 R@1 mIoU、R@1 IoU@0.1/0.3/0.5 和 R@5 mIoU、R@5 IoU@0.1/0.3/0.5
"""
import json
import numpy as np
from pathlib import Path


def load_json_data(json_path):
    """加载 JSON 数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_metrics_from_proposals(json_data, iou_thresholds=(0.1, 0.3, 0.5)):
    """
    从 JSON 数据中计算指标
    
    参数:
        json_data: 从 AncPropInfo1.json 加载的数据
        iou_thresholds: IoU 阈值列表
    
    返回:
        metrics_dict: 包含所有指标的字典
    """
    samples = json_data['samples']
    total_samples = len(samples)
    
    # 初始化指标
    metrics = {
        'r@1_miou': [],
        'r@5_miou': [],
    }
    for thr in iou_thresholds:
        metrics[f'r@1_iou{thr}'] = 0
        metrics[f'r@5_iou{thr}'] = 0
    
    # 遍历每个样本
    for sample in samples:
        proposals = sample['proposals']
        
        # 按 weighted_nll_loss 排序（从小到大）
        sorted_props = sorted(proposals, key=lambda x: x['weighted_nll_loss'])
        
        # R@1 指标
        top1_iou = sorted_props[0]['iou']
        metrics['r@1_miou'].append(top1_iou)
        
        for thr in iou_thresholds:
            if top1_iou >= thr:
                metrics[f'r@1_iou{thr}'] += 1
        
        # R@5 指标
        top5 = sorted_props[:5]
        top5_best_iou = max([p['iou'] for p in top5])
        metrics['r@5_miou'].append(top5_best_iou)
        
        for thr in iou_thresholds:
            if top5_best_iou >= thr:
                metrics[f'r@5_iou{thr}'] += 1
    
    # 计算平均值和比例
    result = {
        'r@1_miou': np.mean(metrics['r@1_miou']),
        'r@5_miou': np.mean(metrics['r@5_miou']),
    }
    
    for thr in iou_thresholds:
        result[f'r@1_iou{thr}'] = metrics[f'r@1_iou{thr}'] / total_samples
        result[f'r@5_iou{thr}'] = metrics[f'r@5_iou{thr}'] / total_samples
    
    result['total_samples'] = total_samples
    
    return result


def main():
    json_path = Path(__file__).parent / 'AncPropInfo1.json'
    
    if not json_path.exists():
        print(f"Error: {json_path} not found")
        return
    
    print(f"Loading data from {json_path}...")
    json_data = load_json_data(json_path)
    
    print(f"Computing metrics...")
    metrics = compute_metrics_from_proposals(json_data)
    
    print("\n" + "="*60)
    print("Metrics computed from AncPropInfo1.json (sorted by weighted_nll_loss)")
    print("="*60)
    print(f"Total samples: {metrics['total_samples']}")
    print()
    print(f"R@1 mIoU:     {metrics['r@1_miou']:.4f}")
    print(f"R@1 IoU@0.1:  {metrics['r@1_iou0.1']:.4f}")
    print(f"R@1 IoU@0.3:  {metrics['r@1_iou0.3']:.4f}")
    print(f"R@1 IoU@0.5:  {metrics['r@1_iou0.5']:.4f}")
    print()
    print(f"R@5 mIoU:     {metrics['r@5_miou']:.4f}")
    print(f"R@5 IoU@0.1:  {metrics['r@5_iou0.1']:.4f}")
    print(f"R@5 IoU@0.3:  {metrics['r@5_iou0.3']:.4f}")
    print(f"R@5 IoU@0.5:  {metrics['r@5_iou0.5']:.4f}")
    print("="*60)
    
    # 保存结果到 JSON
    output_path = json_path.parent / 'metrics_from_AncPropInfo1.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {output_path}")


if __name__ == '__main__':
    main()
