import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp


# https://github.com/CoinCheung/pytorch-loss
##
# version 2: user derived grad computation
class FocalSigmoidLossFuncV2(torch.autograd.Function):
    """
    compute backward directly for better numeric stability
    """

    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, logits, label, alpha, gamma):
        #  logits = logits.float()

        probs = torch.sigmoid(logits)
        coeff = (label - probs).abs_().pow_(gamma).neg_()
        log_probs = torch.where(logits >= 0,
                                F.softplus(logits, -1, 20),
                                logits - F.softplus(logits, 1, 20))
        log_1_probs = torch.where(logits >= 0,
                                  -logits + F.softplus(logits, -1, 20),
                                  -F.softplus(logits, 1, 20))
        ce_term1 = log_probs.mul_(label).mul_(alpha)
        ce_term2 = log_1_probs.mul_(1. - label).mul_(1. - alpha)
        ce = ce_term1.add_(ce_term2)
        loss = ce * coeff

        ctx.vars = (coeff, probs, ce, label, gamma, alpha)

        return loss

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        """
        compute gradient of focal loss
        """
        (coeff, probs, ce, label, gamma, alpha) = ctx.vars

        d_coeff = (label - probs).abs_().pow_(gamma - 1.).mul_(gamma)
        d_coeff.mul_(probs).mul_(1. - probs)
        d_coeff = torch.where(label < probs, d_coeff.neg(), d_coeff)
        term1 = d_coeff.mul_(ce)

        d_ce = label * alpha
        d_ce.sub_(probs.mul_((label * alpha).mul_(2).add_(1).sub_(label).sub_(alpha)))
        term2 = d_ce.mul(coeff)

        grads = term1.add_(term2)
        grads.mul_(grad_output)

        return grads, None, None, None


class FocalLossV2(nn.Module):
    """
    平衡正负样本和难易样本
    """

    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean'):
        super(FocalLossV2, self).__init__()
        self.alpha = alpha  # 平衡正负样本的权重
        self.gamma = gamma  # 调整难易样本的权重
        self.reduction = reduction  # 输出损失的方法

    def forward(self, logits, label):
        """

        :param logits:
        :param label:
        :return:

        examples:
            Usage is same as nn.BCEWithLogits:
            >>> criteria = FocalLossV2()
            >>> logits = torch.randn(8, 19, 384, 384)
            >>> lbs = torch.randint(0, 2, (8, 19, 384, 384)).float()
            >>> loss = criteria(logits, lbs)
        """
        loss = FocalSigmoidLossFuncV2.apply(logits, label, self.alpha, self.gamma)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


class SimilarityLoss(nn.Module):
    def __init__(self, alpha=1.0, delta=1.0, normalize=True):
        """
        :param alpha: 余弦相似度的权重
        :param delta: KL-Divergence 的权重
        :param normalize: 是否对输入特征进行归一化
        """
        super(SimilarityLoss, self).__init__()
        self.alpha = alpha
        self.delta = delta
        self.normalize = normalize

    def forward(self, pair):
        P, Q = pair
        cosine_loss = self.cosine_similarity_loss(P, Q)
        kl_loss = self.kl_divergence_loss(P, Q)
        return self.alpha * cosine_loss + self.delta * kl_loss

    def cosine_similarity_loss(self, P, Q):
        # P 和 Q 应该是 [batch_size, ...] 的特征图
        P_flat = P.reshape(P.size(0), -1)
        Q_flat = Q.reshape(Q.size(0), -1)
        if self.normalize:
            # 可选归一化（例如 L2 归一化）; 这里也可以选择 F.normalize
            P_flat = F.normalize(P_flat, p=2, dim=1)
            Q_flat = F.normalize(Q_flat, p=2, dim=1)
        cos_sim = F.cosine_similarity(P_flat, Q_flat, dim=-1)
        return 1 - cos_sim.mean()

    def kl_divergence_loss(self, P, Q):
        # 对于 KL-Divergence，我们需要将特征视为概率分布，因此使用 softmax
        P_flat = P.reshape(P.size(0), -1)
        Q_flat = Q.reshape(Q.size(0), -1)
        if self.normalize:
            # 使用 softmax 转换为概率分布（在每个样本上归一化）
            P_prob = F.softmax(P_flat, dim=1)
            Q_prob = F.softmax(Q_flat, dim=1)
        else:
            P_prob = P_flat
            Q_prob = Q_flat
        epsilon = 1e-10
        P_prob = P_prob + epsilon
        Q_prob = Q_prob + epsilon
        # 计算每个样本的双向 KL-Divergence，然后取平均
        kl_loss = 0
        for i in range(P_prob.size(0)):
            kl_p_q = torch.sum(P_prob[i] * torch.log(P_prob[i] / Q_prob[i]))
            kl_q_p = torch.sum(Q_prob[i] * torch.log(Q_prob[i] / P_prob[i]))
            kl_loss += (kl_p_q + kl_q_p)
        return kl_loss / P_prob.size(0)


class WeightBCEWithLogitsLoss(nn.Module):
    def __init__(self, neg_weight=None, pos_weight=None, reduction='mean'):
        """
        自定义的 BCEWithLogitsLoss，支持正负样本权重

        Args:
            neg_weight (Tensor, optional): 负样本的权重（作用于 (1-y)log(1-sigma(x)) 项）
            pos_weight (Tensor, optional): 正样本的权重（作用于 y log(sigma(x)) 项）
            reduction (str): 'mean'（默认）、'sum' 或 'none'
        """
        super().__init__()
        self.neg_weight = self._validate_weight(neg_weight, "neg_weight")
        self.pos_weight = self._validate_weight(pos_weight, "pos_weight")
        self.reduction = reduction

    def _validate_weight(self, weight, name):
        """验证权重参数的合法性，并转换为合适的格式"""
        if weight is None or str(weight).lower() == 'none':
            return None

        try:
            # 尝试转换为 float（支持整数、小数、字符串形式的数字）
            weight_float = float(weight)
            if weight_float < 0:
                raise ValueError(f"{name} 不能为负数，但输入为 {weight}")
            return torch.tensor(weight_float, dtype=torch.float32)
        except (ValueError, TypeError):
            raise ValueError(
                f"{name} 必须是 None、数值或 'None' 字符串，但输入为 {weight}（类型: {type(weight)}）"
            )

    def forward(self, logits, target):
        """
        logits: 模型输出的原始 logits（未经过 Sigmoid）
        target: 目标值（0 或 1）
        """
        # 计算基础损失（数值稳定版本）
        if self.pos_weight is not None:
            loss = - (self.pos_weight * target * F.logsigmoid(logits)) - ((1 - target) * F.logsigmoid(-logits))
        else:
            loss = - (target * F.logsigmoid(logits)) - ((1 - target) * F.logsigmoid(-logits))

        # 应用负样本权重
        if self.neg_weight is not None:
            loss = - (target * F.logsigmoid(logits)) - (self.neg_weight * (1 - target) * F.logsigmoid(-logits))

        # 根据 reduction 返回结果
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def extra_repr(self):
        # 打印类参数信息（仿照 PyTorch 原生风格）
        s = f"neg_weight={self.neg_weight}, pos_weight={self.pos_weight}"
        if self.reduction != 'mean':
            s += f", reduction='{self.reduction}'"
        return s
