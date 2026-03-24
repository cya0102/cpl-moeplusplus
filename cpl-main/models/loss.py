import torch
import torch.nn.functional as F
import pdb


def cal_nll_loss(logit, idx, mask, weights=None):
    eps = 0.1
    acc = (logit.max(dim=-1)[1]==idx).float()
    mean_acc = (acc * mask).sum() / mask.sum()
    
    logit = logit.log_softmax(dim=-1)
    nll_loss = -logit.gather(dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)
    smooth_loss = -logit.sum(dim=-1)
    nll_loss = (1 - eps) * nll_loss + eps / logit.size(-1) * smooth_loss
    if weights is None:
        nll_loss = nll_loss.masked_fill(mask == 0, 0)
        nll_loss = nll_loss.sum(dim=-1) / mask.sum(dim=-1)
    else:
        nll_loss = (nll_loss * weights).sum(dim=-1)

    return nll_loss.contiguous(), mean_acc


def cal_nll_loss_with_width(logit, idx, mask, width, weights=None):
    """
    计算交叉熵损失，并返回标准化长度信息用于后续排序
    
    参数:
        logit: 模型输出的logit
        idx: 目标词索引
        mask: 掩码
        width: 提议的宽度（已归一化到[0,1]）
        weights: 可选的权重
    
    返回:
        nll_loss: 交叉熵损失
        mean_acc: 准确率
        width: 标准化长度（即宽度）
    """
    eps = 0.1
    acc = (logit.max(dim=-1)[1]==idx).float()
    mean_acc = (acc * mask).sum() / mask.sum()
    
    logit = logit.log_softmax(dim=-1)
    nll_loss = -logit.gather(dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)
    smooth_loss = -logit.sum(dim=-1)
    nll_loss = (1 - eps) * nll_loss + eps / logit.size(-1) * smooth_loss
    if weights is None:
        nll_loss = nll_loss.masked_fill(mask == 0, 0)
        nll_loss = nll_loss.sum(dim=-1) / mask.sum(dim=-1)
    else:
        nll_loss = (nll_loss * weights).sum(dim=-1)

    return nll_loss.contiguous(), mean_acc, width


def rec_loss(words_logit, words_id, words_mask, num_props, ref_words_logit=None, **kwargs):
    bsz = words_logit.size(0) // num_props
    words_mask1 = words_mask.unsqueeze(1) \
        .expand(bsz, num_props, -1).contiguous().view(bsz*num_props, -1)
    words_id1 = words_id.unsqueeze(1) \
        .expand(bsz, num_props, -1).contiguous().view(bsz*num_props, -1)

    nll_loss, acc = cal_nll_loss(words_logit, words_id1, words_mask1)
    nll_loss = nll_loss.view(bsz, num_props)
    min_nll_loss = nll_loss.min(dim=-1)[0]

    final_loss = min_nll_loss.mean()

    if ref_words_logit is not None:
        ref_nll_loss, ref_acc = cal_nll_loss(ref_words_logit, words_id, words_mask) 
        final_loss = final_loss + ref_nll_loss.mean()
        final_loss = final_loss / 2
    
    loss_dict = {
        'final_loss': final_loss.item(),
        'nll_loss': min_nll_loss.mean().item(),
    }
    if ref_words_logit is not None:
        loss_dict.update({
            'ref_nll_loss': ref_nll_loss.mean().item(),
            })

    return final_loss, loss_dict

    
def ivc_loss(words_logit, words_id, words_mask, num_props, neg_words_logit_1=None, neg_words_logit_2=None, ref_words_logit=None, **kwargs):
    bsz = words_logit.size(0) // num_props
    words_mask1 = words_mask.unsqueeze(1) \
        .expand(bsz, num_props, -1).contiguous().view(bsz*num_props, -1)
    words_id1 = words_id.unsqueeze(1) \
        .expand(bsz, num_props, -1).contiguous().view(bsz*num_props, -1)

    nll_loss, acc = cal_nll_loss(words_logit, words_id1, words_mask1)
    min_nll_loss, idx = nll_loss.view(bsz, num_props).min(dim=-1)

    if ref_words_logit is not None:
        ref_nll_loss, ref_acc = cal_nll_loss(ref_words_logit, words_id, words_mask)
        tmp_0 = torch.zeros_like(min_nll_loss).cuda()
        tmp_0.requires_grad = False
        ref_loss = torch.max(min_nll_loss - ref_nll_loss + kwargs["margin_1"], tmp_0)
        rank_loss = ref_loss.mean()
    else:
        rank_loss = min_nll_loss.mean()

    width_tensor = kwargs.get('width')
    use_width_reg = kwargs.get('use_width_reg', False)
    width_lambda = kwargs.get('width_lambda', 0.0)
    width_loss = words_logit.new_tensor(0.0)
    if use_width_reg and width_tensor is not None:
        width_tensor = width_tensor.view(bsz, num_props)
        selected_width = width_tensor.gather(index=idx.unsqueeze(-1), dim=-1).squeeze(-1)
        width_loss = selected_width.mean()
        # width_loss = torch.clamp(selected_width - 0.05, min=0).mean()  # Margin guard for width collapse.
    
    if neg_words_logit_1 is not None:
        neg_nll_loss_1, neg_acc_1 = cal_nll_loss(neg_words_logit_1, words_id1, words_mask1)
        neg_nll_loss_1 = torch.gather(neg_nll_loss_1.view(bsz, num_props), index=idx.unsqueeze(-1), dim=-1).squeeze(-1)
        tmp_0 = torch.zeros_like(min_nll_loss).cuda()
        tmp_0.requires_grad = False
        neg_loss_1 = torch.max(min_nll_loss - neg_nll_loss_1 + kwargs["margin_2"], tmp_0)
        rank_loss = rank_loss + neg_loss_1.mean()
    
    if neg_words_logit_2 is not None:
        neg_nll_loss_2, neg_acc_2 = cal_nll_loss(neg_words_logit_2, words_id1, words_mask1)
        neg_nll_loss_2 = torch.gather(neg_nll_loss_2.view(bsz, num_props), index=idx.unsqueeze(-1), dim=-1).squeeze(-1)
        tmp_0 = torch.zeros_like(min_nll_loss).cuda()
        tmp_0.requires_grad = False
        neg_loss_2 = torch.max(min_nll_loss - neg_nll_loss_2 + kwargs["margin_2"], tmp_0)
        rank_loss = rank_loss + neg_loss_2.mean()

    loss = kwargs['alpha_1'] * rank_loss

    gauss_weight = kwargs['gauss_weight'].view(bsz, num_props, -1)
    gauss_weight = gauss_weight / gauss_weight.sum(dim=-1, keepdim=True)
    target = torch.eye(num_props).unsqueeze(0).cuda() * kwargs["lambda"]
    source = torch.matmul(gauss_weight, gauss_weight.transpose(1, 2))
    div_loss = torch.norm(target - source, dim=(1, 2))**2

    loss = loss + kwargs['alpha_2'] * div_loss.mean()
    width_penalty = width_lambda * width_loss
    loss = loss + width_penalty

    return loss, {
        'ivc_loss': loss.item(),
        'neg_loss_1': neg_loss_1.mean().item() if neg_words_logit_1 is not None else 0.0,
        'neg_loss_2': neg_loss_2.mean().item() if neg_words_logit_2 is not None else 0.0,
        'ref_loss': ref_loss.mean().item() if ref_words_logit is not None else 0.0,
        'div_loss': div_loss.mean().item(),
        'width_loss': width_penalty.item()
    }
