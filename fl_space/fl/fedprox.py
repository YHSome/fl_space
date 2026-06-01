"""
FedProx 算法 — 联邦近端优化 (Federated Proximal)

论文: "Federated Optimization in Heterogeneous Networks"
      (Li et al., MLSys 2020)

与 FedAvg 的关键差异：
    - 本地训练时在损失函数中加入 proximal term:
      L_prox = L(w) + (μ/2) * ||w - w_global||²
    - 这约束了本地更新不要偏离全局模型太远，
      有效处理数据/系统异构性问题。
    - μ = 0 时退化为标准 FedAvg。

组件复用：
    客户端选择器 → 复用 FedAvg 的 RandomSelector
    聚合器       → 复用 FedAvg 的 SyncWeightedAggregator
    评估器       → 复用 FedAvg 的 StandardEvaluator
    本地训练器   → 新增 ProximalTrainer（仅此不同）
"""

from __future__ import annotations

import copy
from typing import Any

from fl_space.fl.core import (
    Aggregator,
    ClientSelector,
    ClientUpdate,
    Evaluator,
    LocalTrainer,
)
from fl_space.fl.fedavg import (
    RandomSelector,
    StandardEvaluator,
    SyncWeightedAggregator,
)

# PyTorch 可选依赖
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ── FedProx 特有的本地训练器 ─────────────────────────────────


class ProximalTrainer(LocalTrainer):
    """
    FedProx 本地训练器。

    在标准交叉熵损失上添加 proximal term:
        L_total = L_CE(w) + (μ/2) * ||w - w_global||²

    这使得本地更新不会因为数据异构而偏离全局模型太远。

    Parameters
    ----------
    local_epochs : int
        每轮本地训练的 epoch 数，默认 5。
    batch_size : int
        训练 batch size，默认 32。
    learning_rate : float
        学习率，默认 0.01。
    mu : float
        近端项系数 μ。μ 越大，本地更新越接近全局模型。
        μ = 0 退化为 FedAvg。
    device : str
        计算设备，默认 "cpu"。
    """

    def __init__(
        self,
        local_epochs: int = 5,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        mu: float = 0.01,
        device: str = "cpu",
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("FedProx 需要 PyTorch，请运行: pip install torch")

        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.mu = mu
        self.device = device

    def train(
        self,
        client_id: int,
        model: nn.Module,
        train_loader: DataLoader,
        global_weights: list[Any],
        round_num: int,
        **kwargs: Any,
    ) -> ClientUpdate:
        """
        执行 FedProx 本地训练（含 proximal term）。
        """
        local_model = copy.deepcopy(model)
        local_model.to(self.device)
        local_model.train()

        # 保存全局参数作为 proximal term 的锚点
        global_params = [w.clone().detach().to(self.device) for w in global_weights]

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            local_model.parameters(),
            lr=self.learning_rate,
        )

        data_size = len(train_loader.dataset)  # type: ignore
        total_loss = 0.0

        for _epoch in range(self.local_epochs):
            epoch_loss = 0.0
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = local_model(data)

                # 标准交叉熵损失
                ce_loss = criterion(output, target)

                # Proximal term: (μ/2) * Σ||w_local - w_global||²
                proximal_term = 0.0
                for local_param, global_param in zip(
                    local_model.parameters(), global_params
                ):
                    proximal_term += torch.sum(
                        (local_param - global_param) ** 2
                    )

                loss = ce_loss + (self.mu / 2.0) * proximal_term

                loss.backward()
                optimizer.step()
                epoch_loss += ce_loss.item()  # 记录纯 CE 损失，不含 proximal

            total_loss += epoch_loss

        avg_loss = total_loss / max(self.local_epochs, 1)

        local_weights = [
            param.data.clone() for param in local_model.parameters()
        ]

        return ClientUpdate(
            client_id=client_id,
            weights=local_weights,
            data_size=data_size,
            train_loss=avg_loss,
            round_num=round_num,
        )


# ── 便捷构建函数 ──────────────────────────────────────────────


def create_fedprox_components(
    fraction: float = 0.5,
    min_clients: int = 2,
    local_epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    mu: float = 0.01,
    device: str = "cpu",
    seed: int | None = None,
) -> tuple[ClientSelector, LocalTrainer, Aggregator, Evaluator]:
    """
    一键创建 FedProx 的四件套组件。

    注意：selector、aggregator、evaluator 与 FedAvg 完全相同，
    只有 trainer 使用 ProximalTrainer。

    Parameters
    ----------
    fraction : float
        每轮客户端参与比例。
    min_clients : int
        最少参与客户端数。
    local_epochs : int
        本地训练 epoch 数。
    batch_size : int
        训练 batch size。
    learning_rate : float
        学习率。
    mu : float
        近端项系数。μ=0 即退化为 FedAvg。
    device : str
        计算设备。
    seed : int | None
        随机种子。

    Returns
    -------
    tuple
        (selector, trainer, aggregator, evaluator)

    使用示例::

        from fl_space.fl.fedprox import create_fedprox_components

        selector, trainer, aggregator, evaluator = create_fedprox_components(
            fraction=0.5,
            mu=0.1,  # 更强的近端约束
            device="cuda",
        )
    """
    selector = RandomSelector(
        fraction=fraction,
        min_clients=min_clients,
        seed=seed,
    )
    trainer = ProximalTrainer(
        local_epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        mu=mu,
        device=device,
    )
    aggregator = SyncWeightedAggregator(
        min_updates=max(1, min_clients),
    )
    evaluator = StandardEvaluator(device=device)

    return selector, trainer, aggregator, evaluator
