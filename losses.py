import torch
import torch.nn as nn
import torch.nn.functional as F


class PPLoss(nn.Module):
    def __init__(self, delta=10.0, alpha=1.0, beta=0.1, reduction='mean'):
        """
        Args:
            delta (float): Push Loss的间隔阈值，类间原型距离需大于delta，否则惩罚。
            alpha (float): Pull Loss的权重系数（类内聚合强度）。
            beta (float): Push Loss的权重系数（类间分离强度）。
            reduction (str): 损失计算方式，可选 'mean' 或 'sum'。
        """
        super().__init__()
        self.unique_labels = None
        self.delta = delta
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction

    def forward(self, features, labels, prototypes=None):
        """
        Args:
            features (Tensor): 输入特征向量，形状 [B, D]
            labels (Tensor): 类别标签，形状 [B]
            prototypes (Tensor, optional): 外部传入的类别原型矩阵，形状 [C, D]

        Returns:
            total_loss (Tensor): 总损失（Pull + Push）
            pull_loss (Tensor): 类内聚合损失
            push_loss (Tensor): 类间分离损失
        """
        # 如果未传入外部原型，则动态计算当前batch的类别原型
        if prototypes is None:
            prototypes = self._compute_prototypes(features, labels)

        # 处理labels
        unique_labels = torch.unique(labels)
        self.unique_labels = unique_labels

        # 计算Pull Loss：类内聚合（同类特征靠近原型）
        pull_loss = self._compute_pull_loss(features, labels, prototypes)

        # 计算Push Loss：类间分离（不同类原型间距大于delta）
        push_loss = self._compute_push_loss(prototypes)

        # 总损失加权和
        total_loss = self.alpha * pull_loss + self.beta * push_loss

        return total_loss

    def _compute_prototypes(self, features, labels):
        """
        动态计算当前batch中每个类别的原型（均值）
        形状：
            features: [B, D]
            labels: [B]
        Returns:
            prototypes: [C_batch, D]，C_batch为当前batch中实际存在的类别数
        """
        prototypes = []
        for c in self.unique_labels:
            mask = (labels == c)
            class_features = features[mask]
            proto = class_features.mean(dim=0)  # [D]
            prototypes.append(proto)
        return torch.stack(prototypes, dim=0)  # [C_batch, D]

    def _compute_pull_loss(self, features, labels, prototypes):
        """
        计算类内聚合损失：所有样本特征与对应类别原型的平均距离（固定35类）
        Args:
            features: [B, 768]
            labels: [B]（标签范围0~34）
            prototypes: [35, 768]
        """
        # 根据标签索引对应的原型 [B, 768]
        target_prototypes = prototypes[labels]  # 直接通过labels索引

        # 计算每个样本特征与原型的平方距离 [B]
        distances = torch.sum((features - target_prototypes) ** 2, dim=1)

        # 平均距离作为Pull Loss
        pull_loss = torch.mean(distances)
        return pull_loss

    def _compute_push_loss(self, prototypes):
        """
        计算类间分离损失：强制35个类原型间距大于delta
        Args:
            prototypes: [35, 768]
        """
        # 计算类别数
        num_classes = self.unique_labels.size(0)

        # 计算所有原型对的欧氏距离 [num_classes, num_classes]
        pairwise_dist = torch.cdist(prototypes, prototypes, p=2)

        # 排除对角线（自身距离）
        mask = 1 - torch.eye(num_classes, device=prototypes.device)
        penalty = torch.relu(self.delta - pairwise_dist * mask)  # [num_classes, num_classes]

        # 仅计算上三角部分（避免重复计算）
        upper_tri = torch.triu_indices(num_classes, num_classes, offset=1)
        push_loss = torch.mean(penalty[upper_tri[0], upper_tri[1]] ** 2)

        return push_loss

    def extra_repr(self):
        return f"delta={self.delta}, alpha={self.alpha}, beta={self.beta}"


class NCMLoss(nn.Module):
    def __init__(self, temperature=1.0, epsilon=1e-8):
        """
        Args:
            temperature (float): 温度系数，用于缩放距离影响（类似对比学习）
            epsilon (float): 数值稳定项，防止除零错误
        """
        super().__init__()
        self.temperature = temperature
        self.epsilon = epsilon

    def forward(self, features, labels, prototypes=None):
        """
        Args:
            features: 输入特征 [B, D]
            labels: 真实标签 [B]
            prototypes: 可选，预定义类别原型矩阵 [C, D]

        Returns:
            loss: NCM分类损失
        """
        # 动态计算原型（若未提供）
        if prototypes is None:
            prototypes = self._compute_prototypes(features, labels)  # [C_batch, D]

        # 计算特征与所有原型的距离 [B, C]
        distances = torch.cdist(features, prototypes, p=2)  # 欧氏距离

        # 将距离转换为概率（距离越小概率越高）
        logits = -distances / self.temperature  # [B, C]

        # 计算交叉熵损失（需对齐原型索引与标签）
        unique_labels = torch.unique(labels)
        label_mapping = {l.item(): idx for idx, l in enumerate(unique_labels)}
        mapped_labels = torch.tensor([label_mapping[l.item()] for l in labels],
                                     device=features.device)

        loss = F.cross_entropy(logits, mapped_labels)
        return loss

    def _compute_prototypes(self, features, labels):
        """
        根据当前批次动态计算每个类别的原型（均值）
        Args:
            features: [B, D]
            labels: [B]
        Returns:
            prototypes: [C_batch, D]
        """
        unique_labels = torch.unique(labels)
        prototypes = []
        for c in unique_labels:
            mask = (labels == c)
            if mask.sum() == 0:  # 防御：跳过空类别
                continue
            class_features = features[mask]
            proto = class_features.mean(dim=0)  # [D]
            prototypes.append(proto)

        # 处理全空情况（理论上不会发生）
        if len(prototypes) == 0:
            return torch.randn(1, features.size(1), device=features.device)

        return torch.stack(prototypes, dim=0)  # [C_batch, D]

    def extra_repr(self):
        return f"temperature={self.temperature}, epsilon={self.epsilon}"