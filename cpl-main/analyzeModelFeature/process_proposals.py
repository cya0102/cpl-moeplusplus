import json
import argparse


def process_proposals(input_json_path, output_json_path):
    """
    处理 AncPropInfo.json：
    1. 计算每个提议的 weighted_nll_loss = nll_loss * pred_length_norm
    2. 根据 weighted_nll_loss 重新排序每个视频的提议
    3. 保存为新的 JSON 文件
    """
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = data['samples']
    
    # 处理每个样本
    for sample in samples:
        proposals = sample['proposals']
        
        # 计算 weighted_nll_loss 并添加到每个 proposal
        for proposal in proposals:
            nll_loss = proposal['nll_loss']
            pred_length_norm = proposal['pred_length_norm']
            weighted_nll_loss = nll_loss * pred_length_norm
            proposal['weighted_nll_loss'] = weighted_nll_loss
        
        # 根据 weighted_nll_loss 从小到大重新排序
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
        
        # 更新 weighted_nll_loss_margin_1_vs_2
        if len(proposals_sorted) > 1:
            sample['weighted_nll_loss_margin_1_vs_2'] = (
                proposals_sorted[0]['weighted_nll_loss'] - 
                proposals_sorted[1]['weighted_nll_loss']
            )
        else:
            sample['weighted_nll_loss_margin_1_vs_2'] = None
    
    # 保存结果
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Processing complete. Results saved to {output_json_path}")
    print(f"Total samples processed: {len(samples)}")


def main():
    parser = argparse.ArgumentParser(
        description="Process proposals by computing weighted_nll_loss and re-ranking"
    )
    parser.add_argument('--input', type=str, default='AncPropInfo.json',
                        help='Path to input AncPropInfo.json')
    parser.add_argument('--output', type=str, default='AncPropInfo1.json',
                        help='Path to output JSON file')
    
    args = parser.parse_args()
    
    process_proposals(args.input, args.output)


if __name__ == '__main__':
    main()
