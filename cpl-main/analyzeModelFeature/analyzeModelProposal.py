import torch
import torch.nn.functional as F
import json
import numpy as np
from tqdm import tqdm
import os

def analyze_model_predictions(model, dataloader, device, output_file="error_analysis.json"):
    """
    运行推理，记录每个视频的 Top-5 预测区间和得分，分析 Rank1 掉点的具体原因。
    """
    model.eval()
    analysis_results = []

    # 统计指标
    rank1_shorter_than_gt = 0
    rank1_longer_than_gt = 0
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Analyzing Predictions")):
            # 这里假设你的 dataloader 返回的数据字典格式如下，请根据实际情况微调
            frames_feat = batch_data['frames_feat'].to(device)
            frames_len = batch_data['frames_len'].to(device)
            words_id = batch_data['words_id'].to(device)
            words_feat = batch_data['words_feat'].to(device)
            words_len = batch_data['words_len'].to(device)
            weights = batch_data['weights'].to(device)
            
            vid_names = batch_data['vid_name'] # 视频ID
            gt_windows = batch_data['gt_window'] # 真实的 [start, end]
            video_durations = batch_data['duration'] # 视频总时长
            
            bsz = frames_feat.shape[0]

            # 1. 前向传播获取预测结果
            outputs = model(frames_feat=frames_feat, frames_len=frames_len, 
                            words_id=words_id, words_feat=words_feat, 
                            words_len=words_len, weights=weights, epoch=100) # epoch参数用于绕过负样本挖掘逻辑

            words_logit = outputs['words_logit'] # [bsz * num_props, seq_len, vocab_size]
            center = outputs['center']           # [bsz * num_props]
            width = outputs['width']             # [bsz * num_props]
            
            # 2. 计算每个 Proposal 的得分 (通常是重构 Query 的似然度)
            # words_id 扩展到对应的维度: [bsz * num_props, seq_len]
            words_id_expanded = words_id.unsqueeze(1).expand(bsz, model.num_props, -1).contiguous().view(bsz * model.num_props, -1)
            
            # 计算 log probability
            log_probs = F.log_softmax(words_logit, dim=-1)
            
            # 取出真实单词对应的 log_prob 并求和作为该 proposal 的得分
            # 这里为了简单，我们计算所有非 padding 词的 log_prob 之和
            mask = (words_id_expanded != 0).float()
            gather_log_probs = torch.gather(log_probs, 2, words_id_expanded.unsqueeze(2)).squeeze(2)
            proposal_scores = (gather_log_probs * mask).sum(dim=1) # [bsz * num_props]
            
            proposal_scores = proposal_scores.view(bsz, model.num_props)
            centers = center.view(bsz, model.num_props)
            widths = width.view(bsz, model.num_props)

            # 3. 对每个 Batch 的样本进行单独分析
            for i in range(bsz):
                scores_i = proposal_scores[i]
                centers_i = centers[i]
                widths_i = widths[i]
                
                # 按得分排序，获取 Top-5
                top_scores, top_indices = torch.topk(scores_i, k=5, largest=True)
                
                top5_preds = []
                duration = float(video_durations[i])
                gt_start, gt_end = float(gt_windows[i][0]), float(gt_windows[i][1])
                gt_len = gt_end - gt_start
                
                for rank, idx in enumerate(top_indices):
                    c = centers_i[idx].item() * duration
                    w = widths_i[idx].item() * duration
                    pred_start = max(0.0, c - w / 2)
                    pred_end = min(duration, c + w / 2)
                    
                    # 计算 IoU
                    inter_start = max(gt_start, pred_start)
                    inter_end = min(gt_end, pred_end)
                    inter_len = max(0.0, inter_end - inter_start)
                    union_len = (pred_end - pred_start) + (gt_end - gt_start) - inter_len
                    iou = inter_len / union_len if union_len > 0 else 0.0
                    
                    top5_preds.append({
                        "rank": rank + 1,
                        "score": float(top_scores[rank].item()),
                        "pred_window": [pred_start, pred_end],
                        "iou": iou
                    })
                
                # 分析 Rank1 错在哪
                rank1_pred = top5_preds[0]['pred_window']
                rank1_len = rank1_pred[1] - rank1_pred[0]
                if rank1_len < gt_len * 0.5:
                    rank1_shorter_than_gt += 1
                elif rank1_len > gt_len * 1.5:
                    rank1_longer_than_gt += 1

                analysis_results.append({
                    "vid_name": vid_names[i],
                    "duration": duration,
                    "gt_window": [gt_start, gt_end],
                    "top5_predictions": top5_preds,
                    "score_margin_1_vs_2": top5_preds[0]['score'] - top5_preds[1]['score'] # 记录Top1和Top2的分差
                })

    # 保存结果
    with open(output_file, 'w') as f:
        json.append(analysis_results, f, indent=4)
    
    print(f"Analysis complete. Results saved to {output_file}")
    print(f"Stats: Rank1 is much shorter than GT in {rank1_shorter_than_gt} cases.")
    print(f"Stats: Rank1 is much longer than GT in {rank1_longer_than_gt} cases.")

# 调用示例 (你需要根据你的实际启动脚本调整传入的变量):
# analyze_model_predictions(model, test_loader, device='cuda')