import torch
import torch.nn.functional as F
from copy import deepcopy



def construct_positive_negative_pairs(hs, rs, ts, num_negative_samples=6, use_hard_negatives=True):
    """
    构造正负样本对
    Args:
        hs: [N, D] 头实体的嵌入 (Head)
        rs: [N, D] 关系的嵌入 (Relation)
        ts: [N, D] 尾实体的嵌入 (Tail)
        num_negative_samples: 每个正样本采样的负样本数量
        use_hard_negatives: 是否引入困难负样本

    Returns:
        positive_pairs: 正样本对的相似度列表
        negative_pairs: 负样本对的相似度列表
    """
    device = hs.device
    num_samples = hs.size(0)

    # 正样本对
    positive_sim = F.cosine_similarity(hs + rs, ts, dim=-1)

    # 负样本对
    negative_sim = []
    for i in range(num_samples):
        # 创建整数索引排除当前样本 i
        neg_indices = torch.arange(num_samples, device=device, dtype=torch.long)
        neg_indices = neg_indices[neg_indices != i]  # 排除当前样本 i
        neg_indices = neg_indices[:min(len(neg_indices), num_negative_samples)]  # 确保切片数量

        # 验证 neg_indices 的类型和形状
        assert neg_indices.dtype in [torch.int32, torch.int64], f"Invalid dtype: {neg_indices.dtype}"
        assert neg_indices.dim() == 1, f"Invalid shape: {neg_indices.shape}"

        # 负样本计算
        neg_hs = hs[neg_indices] + rs[neg_indices]
        neg_ts = ts[i].expand_as(neg_hs)
        sim = F.cosine_similarity(neg_hs, neg_ts, dim=-1)

        # 如果使用困难负样本
        if use_hard_negatives:
            hard_neg_count = max(1, num_negative_samples // 2)
            topk_sim, _ = torch.topk(sim, k=min(hard_neg_count, sim.size(0)))
            sim = topk_sim

        negative_sim.append(sim)

    # 将负样本相似度拼接到一起
    negative_sim = torch.cat(negative_sim, dim=0).to(device)
    return positive_sim, negative_sim



import torch
import torch.nn.functional as F

def compute_contrastive_loss(hs_pos, ts_pos, rs_pos, hs_neg, ts_neg, rs_neg, temperature):
    """
    计算对比学习的 InfoNCE 损失

    Args:
        hs_pos: 正样本头实体嵌入 [N, D]
        ts_pos: 正样本尾实体嵌入 [N, D]
        rs_pos: 正样本关系嵌入 [N, D]
        hs_neg: 负样本头实体嵌入 [M, D]
        ts_neg: 负样本尾实体嵌入 [M, D]
        rs_neg: 负样本关系嵌入 [M, D]
        temperature: 对比学习温度参数

    Returns:
        cl_loss: 对比学习损失
    """
    # 构造正样本对
    positive_sim = F.cosine_similarity(hs_pos + rs_pos, ts_pos, dim=-1)  # [N]

    # 构造负样本对
    num_pos = hs_pos.size(0)  # 正样本数量
    num_neg = hs_neg.size(0)  # 负样本数量

    # 负样本特征扩展
    neg_hs_rs = (hs_neg + rs_neg).unsqueeze(0).expand(num_pos, num_neg, -1)  # [N, M, D]
    neg_ts = ts_neg.unsqueeze(0).expand(num_pos, num_neg, -1)  # [N, M, D]

    # 计算负样本对的相似度
    negative_sim = F.cosine_similarity(neg_hs_rs, neg_ts, dim=-1)  # [N, M]

    # 计算正样本 logits
    temperature = max(temperature, 0.1)  # 确保温度不小于 0.1
    positive_logits = positive_sim / temperature  # [N]

    # 计算负样本 logits
    negative_logits = negative_sim / temperature  # [N, M]

    # 计算负样本 log-sum-exp
    negative_log_sum = torch.logsumexp(negative_logits, dim=-1)  # [N]

    # 计算 InfoNCE 损失
    cl_loss = -torch.mean(positive_logits - negative_log_sum)

    return cl_loss



    
def get_label(args, logits, num_labels=-1):
    if args.loss_type == 'balance_softmax':
        th_logit = torch.zeros_like(logits[..., :1])
    else:
        th_logit = logits[:, 0].unsqueeze(1)
    output = torch.zeros_like(logits).to(logits)
    mask = (logits > th_logit)
    if num_labels > 0:
        top_v, _ = torch.topk(logits, num_labels, dim=1)
        top_v = top_v[:, -1]
        mask = (logits >= top_v.unsqueeze(1)) & mask
    output[mask] = 1.0
    output[:, 0] = (output.sum(1) == 0.).to(logits)
    return output


def get_at_loss(logits, labels):
    """
    ATL
    """
    labels = deepcopy(labels)
    # TH label
    th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
    th_label[:, 0] = 1.0
    labels[:, 0] = 0.0
    p_mask = labels + th_label
    n_mask = 1 - labels
    # Rank positive classes to TH
    logit1 = logits - (1 - p_mask) * 1e30
    loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)
    # Rank TH to negative classes
    logit2 = logits - (1 - n_mask) * 1e30
    loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)
    # Sum two parts
    loss = loss1 + loss2
    loss = loss.mean()
    return loss


def get_balance_loss(logits, labels):
    """
    Balanced Softmax
    """
    y_true = labels
    y_pred = logits
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e30
    y_pred_pos = y_pred - (1 - y_true) * 1e30
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat((y_pred_pos, zeros), dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, axis=-1)
    pos_loss = torch.logsumexp(y_pred_pos, axis=-1)
    loss = neg_loss + pos_loss
    loss = loss.mean()
    return loss


def get_af_loss(logits, labels):
    """
    AFL
    """
    # TH label
    th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
    th_label[:, 0] = 1.0
    labels[:, 0] = 0.0
    n_mask = 1 - labels
    num_ex, num_class = labels.size()

    # Rank each class to threshold class TH
    th_mask = torch.cat(num_class * [logits[:, :1]], dim=1)
    logit_th = torch.cat([logits.unsqueeze(1), 1.0 * th_mask.unsqueeze(1)], dim=1)
    log_probs = F.log_softmax(logit_th, dim=1)
    probs = torch.exp(F.log_softmax(logit_th, dim=1))

    # Probability of relation class to be negative (0)
    prob_0 = probs[:, 1, :]
    prob_0_gamma = torch.pow(prob_0, 1)
    log_prob_1 = log_probs[:, 0, :]

    # Rank TH to negative classes
    logit2 = logits - (1 - n_mask) * 1e30
    rank2 = F.log_softmax(logit2, dim=-1)

    loss1 = - (log_prob_1 * (1 + prob_0_gamma) * labels)
    loss2 = -(rank2 * th_label).sum(1)

    loss = 1.0 * loss1.sum(1).mean() + 1.0 * loss2.mean()
    return loss


def get_sat_loss(logits, labels):
    """
    SAT
    """
    exp_th = torch.exp(logits[:, 0].unsqueeze(dim=1))

    p_prob = torch.exp(logits) / (torch.exp(logits) + exp_th)
    n_prob = exp_th / (exp_th + torch.exp(logits))

    p_num = labels[:, 1:].sum(dim=1)
    n_num = 96 - p_num

    p_item = -torch.log(p_prob + 1e-30) * labels
    p_item = p_item[:, 1:]
    n_item = -torch.log(n_prob + 1e-30) * (1 - labels)
    n_item = n_item[:, 1:]

    p_loss = p_item.sum(1)
    n_loss = n_item.sum(1)
    loss = p_loss + n_loss
    loss = loss.mean()
    return loss


def get_mean_sat_loss(logits, labels):
    exp_th = torch.exp(logits[:, 0].unsqueeze(dim=1))

    p_prob = torch.exp(logits) / (torch.exp(logits) + exp_th)
    n_prob = exp_th / (exp_th + torch.exp(logits))

    p_num = labels[:, 1:].sum(dim=1)
    n_num = 96 - p_num

    p_item = -torch.log(p_prob + 1e-30) * labels
    p_item = p_item[:, 1:]
    n_item = -torch.log(n_prob + 1e-30) * (1 - labels)
    n_item = n_item[:, 1:]

    p_loss = p_item.sum(1) / (p_num + 1e-30)
    n_loss = n_item.sum(1) / (n_num + 1e-30)
    loss = p_loss + n_loss
    loss = loss.mean()
    return loss


def get_relu_sat_loss(logits, labels, m=5):
    """
    HingeABL
    """
    p_num = labels[:, 1:].sum(dim=1)

    p_logits_diff = logits[:, 0].unsqueeze(dim=1) - logits
    p_logits_imp = F.relu(p_logits_diff + m)
    p_logits_imp = p_logits_imp * labels
    p_logits_imp = p_logits_imp[:, 1:]
    p_logits_imp = p_logits_imp / (p_logits_imp.sum(dim=1).unsqueeze(dim=1) + 1e-30)

    n_logits_diff = logits - logits[:, 0].unsqueeze(dim=1)
    n_logits_imp = F.relu(n_logits_diff + m)
    n_logits_imp = n_logits_imp * (1 - labels)
    n_logits_imp = n_logits_imp[:, 1:]
    n_logits_imp = n_logits_imp / (n_logits_imp.sum(dim=1).unsqueeze(dim=1) + 1e-30)

    exp_th = torch.exp(logits[:, 0].unsqueeze(dim=1))   # margin=5

    p_prob = torch.exp(logits) / (torch.exp(logits) + exp_th)
    n_prob = exp_th / (exp_th + torch.exp(logits))

    p_item = -torch.log(p_prob + 1e-30) * labels
    p_item = p_item[:, 1:] * p_logits_imp
    n_item = -torch.log(n_prob + 1e-30) * (1 - labels)
    n_item = n_item[:, 1:] * n_logits_imp

    p_loss = p_item.sum(1)
    n_loss = n_item.sum(1)
    loss = p_loss + n_loss
    loss = loss.mean()
    return loss


def get_margin_loss(logits, labels):
    """
    AML
    """
    # TH label
    th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
    th_label[:, 0] = 1.0
    labels[:, 0] = 0.0

    # p_mask = labels + th_label
    # n_mask = 1 - labels
    p_mask = labels
    n_mask = 1 - labels
    n_mask[:, 0] = 0.0

    # Rank positive classes to TH
    # print('=====>', logits.shape, p_mask.shape)
    logit1 = logits - (1 - p_mask) * 1e30
    loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)

    # Rank TH to negative classes
    logit2 = logits - (1 - n_mask) * 1e30
    loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)

    logit3 = 1 - logits + logits[:, 0].unsqueeze(1)
    loss3 = (F.relu(logit3) * p_mask).sum(1)

    logit4 = 1 + logits - logits[:, 0].unsqueeze(1)
    loss4 = (F.relu(logit4) * n_mask).sum(1)

    # import ipdb; ipdb.set_trace()

    # Sum two parts
    # loss = loss1 + loss2 + loss3 + loss4
    # loss = loss3 + 0.5 * loss4
    loss = loss3 + loss4
    # loss = torch.sum(loss * r_mask) / torch.sum(r_mask)
    loss = loss.mean()
    return loss

def get_improved_aml_loss(logits, labels, initial_margin=1.0, num_hard_negatives=3, alpha=0.3, beta=0.3, smoothing=0.01, current_step=0, decay_rate=0.005):
    """Improved AML Loss with stability adjustments."""
    device = logits.device

    # 动态调整边界
    margin = max(1.0, initial_margin - decay_rate * current_step)

    # 标签平滑
    num_classes = labels.size(1)
    smoothed_labels = labels * (1 - smoothing) + smoothing / num_classes

    # 分离正负样本
    th_label = torch.zeros_like(labels, dtype=torch.float).to(device)
    th_label[:, 0] = 1.0  # 标记 TH 类
    labels[:, 0] = 0.0    # 避免正样本影响 TH 类

    p_mask = labels  # 正样本掩码
    n_mask = 1 - labels  # 负样本掩码
    n_mask[:, 0] = 0.0  # TH 类不算负样本

    # 屏蔽非目标 logits
    logit1 = logits.masked_fill(p_mask == 0, -1e6)  # 只保留正样本
    logit2 = logits.masked_fill(n_mask == 0, -1e6)  # 只保留负样本

    # 稳定计算 log_softmax
    log_probs1 = F.log_softmax(logit1, dim=-1) * p_mask
    log_probs2 = F.log_softmax(logit2, dim=-1) * n_mask

    # 归一化正负样本损失
    loss1 = -log_probs1.sum(1) / (p_mask.sum(1) + 1e-10)  # 正样本损失
    loss2 = -log_probs2.sum(1) / (n_mask.sum(1) + 1e-10)  # 负样本损失

    # Hinge 损失
    logit3 = 1 - logits + logits[:, 0].unsqueeze(1)
    loss3 = (F.relu(logit3 + margin) * p_mask).sum(1) / (p_mask.sum(1) + 1e-10)

    logit4 = 1 + logits - logits[:, 0].unsqueeze(1)
    loss4 = (F.relu(logit4 + margin) * n_mask).sum(1) / (n_mask.sum(1) + 1e-10)

    # 动态权重
    hinge_loss = loss3 + loss4
    base_loss = loss1 + loss2
    loss = alpha * base_loss + beta * hinge_loss

    # 平均化损失
    loss = loss.mean()
    return loss

