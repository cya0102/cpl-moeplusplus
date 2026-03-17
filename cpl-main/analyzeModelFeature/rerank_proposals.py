import json
import argparse


def rerank_proposals_by_weighted_nll_loss(input_json_path, output_json_path=None):
    """
    对 AncPropInfo.json 中的每个视频样本的所有提议进行重新排序。
    
    计算方式：
    - weighted_nll_loss = nll_loss * pred_length_norm
    - 按 weighted_nll_loss 从小到大重新排序（weighted_nll_loss 越小越好）
    
    参数:
        input_json_path: 输入的 AncPropInfo.json 路径
        output_json_path: 输出文件路径（可选）
    """
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = data['samples']
    
    # 处理每个样本
    for sample in samples:
        proposals = sample['proposals']
        
        # 计算每个 proposal 的 weighted_nll_loss
        for proposal in proposals:
            nll_loss = proposal['nll_loss']
            pred_length_norm = proposal['pred_length_norm']
            weighted_nll_loss = nll_loss * pred_length_norm
            proposal['weighted_nll_loss'] = weighted_nll_loss
        
        # 按 weighted_nll_loss 从小到大排序
        proposals_sorted = sorted(proposals, key=lambda p: p['weighted_nll_loss'])
        
        # 更新 rank 字段
        for new_rank, proposal in enumerate(proposals_sorted, 1):
            proposal['rank'] = new_rank
        
        # 更新样本中的 proposals
        sample['proposals'] = proposals_sorted
        
        # 更新 top1 和 top5 的 IoU 信息
        top1 = proposals_sorted[0]
        top5 = proposals_sorted[:5]
        
        sample['top1_iou'] = top1['iou']
        sample['top5_best_iou'] = max([p['iou'] for p in top5]) if len(top5) > 0 else 0.0
        
        # 更新 weighted_nll_loss margin
        if len(proposals_sorted) > 1:
            sample['weighted_nll_loss_margin_1_vs_2'] = (
                proposals_sorted[0]['weighted_nll_loss'] - 
                proposals_sorted[1]['weighted_nll_loss']
            )
        else:
            sample['weighted_nll_loss_margin_1_vs_2'] = None
    
    # 重新计算汇总指标
    iou_thresholds = [0.1, 0.3, 0.5]
    metrics = {f"r@1_iou{thr}": 0 for thr in iou_thresholds}
    metrics.update({f"r@5_iou{thr}": 0 for thr in iou_thresholds})
    
    for sample in samples:
        top1_iou = sample['top1_iou']
        top5_best_iou = sample['top5_best_iou']
        
        for thr in iou_thresholds:
            if top1_iou >= thr:
                metrics[f"r@1_iou{thr}"] += 1
            if top5_best_iou >= thr:
                metrics[f"r@5_iou{thr}"] += 1
    
    num_samples = len(samples)
    summary_metrics = {k: v / num_samples for k, v in metrics.items()}
    
    # 更新汇总信息
    data['summary']['metrics'] = summary_metrics
    data['meta']['description'] = (
        'Proposals reranked by weighted_nll_loss (nll_loss * pred_length_norm). '
        'Each sample contains all proposals sorted by weighted_nll_loss.'
    )
    
    # 保存结果
    if output_json_path:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {output_json_path}")
    
    # 打印性能指标
    print("\n" + "="*60)
    print("Performance Metrics (Reranked by Weighted NLL Loss)")
    print("="*60)
    print(f"Number of samples: {num_samples}")
    for thr in iou_thresholds:
        print(f"R@1 IoU={thr}: {summary_metrics[f'r@1_iou{thr}']:.4f}")
        print(f"R@5 IoU={thr}: {summary_metrics[f'r@5_iou{thr}']:.4f}")
    print("="*60 + "\n")
    
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Rerank proposals by weighted_nll_loss (nll_loss * pred_length_norm)"
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Path to AncPropInfo.json')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save reranked results (optional)')
    
    args = parser.parse_args()
    
    rerank_proposals_by_weighted_nll_loss(args.input, args.output)


if __name__ == '__main__':
    main()
