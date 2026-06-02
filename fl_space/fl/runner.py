"""
FL 实验运行器 — 一站式 FL 实验入口

职责：
    - 加载数据集并分配到客户端
    - 创建模型
    - 组装组件 → FLServer
    - 执行训练 → 返回结果

设计原则：
    - 数据加载与算法逻辑解耦
    - 支持 IID 和 non-IID 数据分布
    - 结果以字典形式返回，便于后续可视化和分析

使用示例::

    from fl_space.fl.runner import FLRunner

    # 使用预设
    runner = FLRunner.from_preset("fedavg", "small", "mnist")
    history = runner.run(verbose=True)

    # 使用自定义配置
    from fl_space.fl.config import fedavg_config
    runner = FLRunner(config=fedavg_config(num_clients=15, local_epochs=10))
    history = runner.run()
"""

from __future__ import annotations

from typing import Any

from fl_space.fl.config import FLConfig, DATASET_PRESETS, get_preset_config
from fl_space.fl.core import (
    Aggregator,
    ClientSelector,
    Evaluator,
    FLRoundResult,
    LocalTrainer,
)
from fl_space.fl.models import get_model
from fl_space.fl.server import FLServer
from fl_space.fl.scheduler import CommunicationScheduler

# PyTorch 可选依赖
try:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset, Subset
    from torchvision import datasets, transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class FLRunner:
    """
    FL 实验运行器。

    负责：数据准备 → 模型创建 → 组件组装 → 训练执行

    Parameters
    ----------
    config : FLConfig
        FL 实验配置。
    selector : ClientSelector
        客户端选择策略。
    trainer : LocalTrainer
        本地训练策略。
    aggregator : Aggregator
        聚合策略。
    evaluator : Evaluator
        评估策略。
    scheduler : CommunicationScheduler | None
        通信调度器（可选）。
    """

    def __init__(
        self,
        config: FLConfig,
        selector: ClientSelector,
        trainer: LocalTrainer,
        aggregator: Aggregator,
        evaluator: Evaluator,
        scheduler: CommunicationScheduler | None = None,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "FLRunner 需要 PyTorch 和 torchvision。"
                "请运行: pip install fl-space[full]"
            )

        self.config = config
        self.selector = selector
        self.trainer = trainer
        self.aggregator = aggregator
        self.evaluator = evaluator
        self.scheduler = scheduler

        # 运行时状态
        self._server: FLServer | None = None
        self._train_loaders: dict[int, DataLoader] = {}
        self._test_loader: DataLoader | None = None
        self._model: Any = None

    @classmethod
    def from_preset(
        cls,
        algorithm: str = "fedavg",
        scale: str = "small",
        dataset: str = "mnist",
        scheduler: CommunicationScheduler | None = None,
        device: str = "cpu",
        **overrides: Any,
    ) -> "FLRunner":
        """
        从预设配置创建 Runner。

        Parameters
        ----------
        algorithm : str
            算法: "fedavg", "fedprox", "fedbuff"。
        scale : str
            规模: "small", "medium", "large"。
        dataset : str
            数据集: "mnist", "fashion_mnist", "cifar10"。
        scheduler : CommunicationScheduler | None
            通信调度器。
        device : str
            计算设备。
        **overrides
            其他覆盖参数。

        Returns
        -------
        FLRunner
            配置好的 Runner 实例。

        使用示例::

            runner = FLRunner.from_preset("fedprox", "medium", "cifar10", device="cuda")
            history = runner.run()
        """
        config = get_preset_config(
            algorithm=algorithm,
            scale=scale,
            dataset=dataset,
            device=device,
            **overrides,
        )

        # 根据算法创建组件
        algo = algorithm.lower()
        if algo == "fedavg":
            from fl_space.fl.fedavg import create_fedavg_components
            components = create_fedavg_components(
                fraction=config.fraction,
                min_clients=max(1, int(config.num_clients * config.fraction)),
                local_epochs=config.local_epochs,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                device=config.device,
                seed=config.seed,
            )
        elif algo == "fedprox":
            from fl_space.fl.fedprox import create_fedprox_components
            components = create_fedprox_components(
                fraction=config.fraction,
                min_clients=max(1, int(config.num_clients * config.fraction)),
                local_epochs=config.local_epochs,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                mu=config.mu,
                device=config.device,
                seed=config.seed,
            )
        elif algo == "fedbuff":
            from fl_space.fl.fedbuff import create_fedbuff_components
            components = create_fedbuff_components(
                min_clients=config.num_clients,
                local_epochs=config.local_epochs,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                buffer_size=config.buffer_size,
                staleness_weight=config.staleness_weight,
                device=config.device,
            )
        else:
            raise ValueError(f"未知算法: '{algo}'")

        return cls(config, *components, scheduler=scheduler)

    # ── 数据准备 ──────────────────────────────────────────────

    def prepare_data(
        self,
        dataset_name: str = "mnist",
        iid: bool = True,
        alpha: float = 0.5,
        data_dir: str = "./data",
    ) -> None:
        """
        加载数据集并分配到客户端。

        支持 IID（均匀随机分配）和 non-IID（Dirichlet 分配）。

        Parameters
        ----------
        dataset_name : str
            数据集名称: "mnist", "fashion_mnist", "cifar10"。
        iid : bool
            True 表示 IID 分配，False 表示 non-IID。
        alpha : float
            Dirichlet 分布参数（仅 non-IID 时有效）。
            alpha 越小分布越不均匀。
        data_dir : str
            数据下载/缓存目录。

        Raises
        ------
        ImportError
            torchvision 未安装时。
        """
        if not TORCH_AVAILABLE:
            raise ImportError("数据加载需要 torchvision")

        dataset_name = dataset_name.lower()

        # 选择数据集和变换
        if dataset_name in ("mnist", "fashion_mnist"):
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])
            ds_cls = (
                datasets.MNIST if dataset_name == "mnist"
                else datasets.FashionMNIST
            )
            train_ds = ds_cls(data_dir, train=True, download=True, transform=transform)
            test_ds = ds_cls(data_dir, train=False, download=True, transform=transform)

        elif dataset_name == "cifar10":
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010),
                ),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010),
                ),
            ])
            train_ds = datasets.CIFAR10(
                data_dir, train=True, download=True, transform=transform_train,
            )
            test_ds = datasets.CIFAR10(
                data_dir, train=False, download=True, transform=transform_test,
            )
        else:
            raise ValueError(f"不支持的数据集: '{dataset_name}'")

        # 分配数据到客户端
        n_clients = self.config.num_clients
        client_data_indices = self._partition_data(
            train_ds, n_clients, iid=iid, alpha=alpha,
        )

        # 创建 DataLoader
        self._train_loaders = {}
        for cid, indices in enumerate(client_data_indices):
            subset = Subset(train_ds, indices)
            self._train_loaders[cid] = DataLoader(
                subset,
                batch_size=self.config.batch_size,
                shuffle=True,
                drop_last=False,
            )

        self._test_loader = DataLoader(
            test_ds,
            batch_size=self.config.batch_size * 2,
            shuffle=False,
        )

    def _partition_data(
        self,
        dataset: Dataset,
        n_clients: int,
        iid: bool = True,
        alpha: float = 0.5,
    ) -> list[list[int]]:
        """
        将数据集划分为 n_clients 份。

        Parameters
        ----------
        dataset : Dataset
            训练数据集。
        n_clients : int
            客户端数量。
        iid : bool
            True 为 IID，False 为 non-IID (Dirichlet)。
        alpha : float
            Dirichlet 参数。

        Returns
        -------
        list[list[int]]
            每个客户端的样本索引列表。
        """
        n_samples = len(dataset)
        targets = np.array([dataset[i][1] for i in range(n_samples)])
        n_classes = len(np.unique(targets))

        if iid:
            # IID：随机打乱后均匀分配
            indices = np.random.permutation(n_samples)
            split_size = n_samples // n_clients
            client_indices = [
                indices[i * split_size:(i + 1) * split_size].tolist()
                for i in range(n_clients)
            ]
            # 余数分配到最后一个客户端
            remainder = indices[n_clients * split_size:]
            if len(remainder) > 0:
                client_indices[-1].extend(remainder.tolist())
        else:
            # Non-IID：Dirichlet 分布
            # 每个类别按 Dirichlet(alpha) 比例分配给各客户端
            client_indices = [[] for _ in range(n_clients)]

            for c in range(n_classes):
                class_indices = np.where(targets == c)[0]
                np.random.shuffle(class_indices)

                # Dirichlet 分配比例
                proportions = np.random.dirichlet(
                    np.repeat(alpha, n_clients)
                )
                proportions = (
                    proportions * len(class_indices)
                ).astype(int)

                # 修正舍入误差
                diff = len(class_indices) - proportions.sum()
                proportions[-1] += diff

                start = 0
                for cid in range(n_clients):
                    end = start + proportions[cid]
                    client_indices[cid].extend(
                        class_indices[start:end].tolist()
                    )
                    start = end

        return client_indices

    # ── 模型创建 ──────────────────────────────────────────────

    def prepare_model(
        self,
        model_name: str = "mlp",
        **model_kwargs: Any,
    ) -> None:
        """
        创建模型实例。

        Parameters
        ----------
        model_name : str
            模型名称: "mlp" 或 "simplecnn"。
        **model_kwargs
            传递给模型构造函数的参数。
        """
        self._model = get_model(model_name, **model_kwargs)

    # ── 执行训练 ──────────────────────────────────────────────

    def run(
        self,
        dataset_name: str = "mnist",
        iid: bool = True,
        alpha: float = 0.5,
        data_dir: str = "./data",
        verbose: bool = True,
    ) -> list[FLRoundResult]:
        """
        执行完整的 FL 训练流程。

        自动完成：数据加载 → 模型创建 → 训练执行。

        Parameters
        ----------
        dataset_name : str
            数据集名称。
        iid : bool
            是否 IID 分配。
        alpha : float
            non-IID Dirichlet 参数。
        data_dir : str
            数据目录。
        verbose : bool
            是否打印进度。

        Returns
        -------
        list[FLRoundResult]
            每轮训练结果。

        使用示例::

            runner = FLRunner.from_preset("fedavg", "small", "mnist")
            history = runner.run(dataset_name="mnist", iid=True, verbose=True)

            # 查看结果
            for r in history:
                print(f"轮次 {r.round_num}: 准确率 {r.eval_metrics['accuracy']}")
        """
        if verbose:
            print(f"=== SpaceFL 实验 ===")
            print(f"  算法: {self.config.algorithm}")
            print(f"  数据集: {dataset_name}")
            print(f"  客户端: {self.config.num_clients}")
            print(f"  轮次: {self.config.num_rounds}")
            print(f"  设备: {self.config.device}")
            print()

        # 1. 数据准备
        if verbose:
            print("[1/3] 加载数据...")
        self.prepare_data(
            dataset_name=dataset_name,
            iid=iid,
            alpha=alpha,
            data_dir=data_dir,
        )

        # 2. 模型准备
        if verbose:
            print("[2/3] 创建模型...")
        ds_preset = DATASET_PRESETS.get(dataset_name, DATASET_PRESETS["mnist"])
        self.prepare_model(
            model_name=ds_preset["model"],
            **ds_preset.get("model_kwargs", {}),
        )

        # 3. 训练
        if verbose:
            print("[3/3] 开始训练...")
            print()

        self._server = FLServer(
            config=self.config,
            selector=self.selector,
            trainer=self.trainer,
            aggregator=self.aggregator,
            evaluator=self.evaluator,
            scheduler=self.scheduler,
        )

        history = self._server.run(
            model=self._model,
            train_loaders=self._train_loaders,
            test_loader=self._test_loader,
            verbose=verbose,
        )

        if verbose:
            print()
            print("=== 训练完成 ===")
            if history:
                final_acc = history[-1].eval_metrics.get("accuracy", 0)
                print(f"  最终准确率: {final_acc:.4f}")
                first_ts = history[0].timeslot_start
                last_ts = history[-1].timeslot
                total_slots = last_ts - first_ts
                # 使用 time_model 显示可读时间
                tm = self._server.time_model
                ts_dur = getattr(tm, "timeslot_duration_min", 1.0)
                time_str = tm.slots_to_display(total_slots, ts_dur) if tm else f"{total_slots} slots"
                print(f"  虚拟时间范围: TS{first_ts} → TS{last_ts} "
                      f"({time_str}, {len(history)} 轮)")
                print(f"  时间模型: {tm.name}")

        return history

    @property
    def history_dict(self) -> list[dict[str, Any]]:
        """训练历史（字典格式，便于 JSON 序列化）。"""
        if self._server is None:
            return []
        return self._server.get_history_dict()
