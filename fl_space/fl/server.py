"""
FL Server — 联邦学习服务器编排器

职责：
    - 组合四个可插拔组件 (selector, trainer, aggregator, evaluator)
    - 编排完整的 FL 训练流程
    - 管理全局模型和客户端状态
    - 记录和返回训练历史

设计原则：
    - 与具体算法解耦：接受任意 ClientSelector/LocalTrainer/Aggregator/Evaluator 组合
    - 与通信方式解耦：通过 scheduler 获取通信状态
    - 单线程模拟异步场景：FedBuff 可通过 _train_client() 独立模拟单客户端训练
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from fl_space.fl.core import (
    Aggregator,
    ClientSelector,
    ClientState,
    ClientUpdate,
    Evaluator,
    FLRoundResult,
    LocalTrainer,
)
from fl_space.fl.scheduler import CommunicationScheduler
from fl_space.fl.time_model import TimeBreakdown, TimeModel, SlotTimeModel

# PyTorch 可选依赖
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class FLConfig:
    """
    FL 实验完整配置。

    Attributes
    ----------
    algorithm : str
        算法名称: "fedavg", "fedprox", "fedbuff"。
    num_rounds : int
        全局训练轮次数。
    num_clients : int
        客户端总数。
    timeslots_per_round : int
        每轮的时间槽数（用于将连续时间离散化为 FL 轮次）。
    fraction : float
        每轮参与的客户端比例（同步算法）。
    local_epochs : int
        本地训练 epoch 数。
    batch_size : int
        训练 batch size。
    learning_rate : float
        学习率。
    mu : float
        FedProx 近端项系数（仅 FedProx 有效）。
    buffer_size : int
        FedBuff 缓冲区大小 K（仅 FedBuff 有效）。
    staleness_weight : bool
        FedBuff 是否启用陈旧度降权。
    device : str
        计算设备。
    seed : int | None
        随机种子。
    """
    algorithm: str = "fedavg"
    num_rounds: int = 50
    num_clients: int = 10
    timeslots_per_round: int = 10
    fraction: float = 0.5
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.01
    mu: float = 0.01
    buffer_size: int = 5
    staleness_weight: bool = False
    device: str = "cpu"
    seed: int | None = None
    # 时间模型配置
    time_model: str = "slot"
    time_model_kwargs: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, config: dict) -> "FLConfig":
        """
        从字典创建 FLConfig。

        自动过滤不存在的字段，仅使用有效键。

        Parameters
        ----------
        config : dict
            配置字典，键对应 FLConfig 字段名。

        Returns
        -------
        FLConfig
            配置实例。

        使用示例::

            config = FLConfig.from_dict({
                "algorithm": "fedprox",
                "num_rounds": 100,
                "mu": 0.1,
                "device": "cuda",
            })
        """
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in config.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_json(cls, filepath: str) -> "FLConfig":
        """
        从 JSON 文件加载 FLConfig。

        Parameters
        ----------
        filepath : str
            JSON 配置文件路径。

        Returns
        -------
        FLConfig
            配置实例。

        使用示例::

            config = FLConfig.from_json("my_experiment.json")
        """
        import json
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_dict(self) -> dict:
        """
        导出为字典（便于 JSON 序列化和保存）。

        Returns
        -------
        dict
            当前配置的字典表示。
        """
        from dataclasses import asdict
        return asdict(self)

    def to_json(self, filepath: str) -> None:
        """
        保存配置为 JSON 文件。

        Parameters
        ----------
        filepath : str
            输出 JSON 文件路径。
        """
        import json
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


class FLServer:
    """
    FL 训练服务器。

    组合四个可插拔组件，编排完整训练流程。

    支持两种运行模式：
        - run_sync()：同步训练（FedAvg / FedProx）
        - run_async()：异步训练（FedBuff）

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
        通信调度器（可选，未提供时假定所有客户端始终可通信）。

    使用示例::

        from fl_space.fl.server import FLServer, FLConfig
        from fl_space.fl.fedavg import create_fedavg_components

        config = FLConfig(algorithm="fedavg", num_rounds=50)
        components = create_fedavg_components()
        server = FLServer(config, *components)

        history = server.run_sync(model, train_loaders, test_loader)
        print(f"最终准确率: {history[-1].eval_metrics['accuracy']}")
    """

    def __init__(
        self,
        config: FLConfig,
        selector: ClientSelector,
        trainer: LocalTrainer,
        aggregator: Aggregator,
        evaluator: Evaluator,
        scheduler: CommunicationScheduler | None = None,
        time_model: TimeModel | None = None,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("FLServer 需要 PyTorch，请运行: pip install torch")

        self.config = config
        self.selector = selector
        self.trainer = trainer
        self.aggregator = aggregator
        self.evaluator = evaluator
        self.scheduler = scheduler

        # 时间模型：显式传入 > 配置构建
        if time_model is not None:
            self.time_model = time_model
        else:
            self.time_model = self._build_time_model()

        # 运行时状态
        self._global_model: Any = None
        self._clients: list[ClientState] = []
        self._history: list[FLRoundResult] = []

    def _build_time_model(self) -> TimeModel:
        """根据 FLConfig 构建时间模型实例。"""
        kwargs = dict(self.config.time_model_kwargs)

        # 自动继承 timeslot 时长
        sim = self.scheduler._sim if self.scheduler is not None else None
        if sim is not None and "timeslot_duration_min" not in kwargs:
            kwargs["timeslot_duration_min"] = getattr(
                sim, "timeslot_duration_min", 1.0
            )

        return TimeModel.create(self.config.time_model, **kwargs)

    @property
    def history(self) -> list[FLRoundResult]:
        """训练历史记录。"""
        return self._history

    @staticmethod
    def _get_client_data_sizes(
        train_loaders: dict[int, Any],
    ) -> dict[int, int]:
        """
        从 DataLoader 提取每个客户端的数据量。

        Parameters
        ----------
        train_loaders : dict[int, DataLoader]
            客户端 ID → DataLoader 映射。

        Returns
        -------
        dict[int, int]
            客户端 ID → 样本数映射。
        """
        sizes: dict[int, int] = {}
        for cid, loader in train_loaders.items():
            try:
                ds = loader.dataset
                if hasattr(ds, "__len__"):
                    sizes[cid] = len(ds)
                else:
                    sizes[cid] = 100
            except Exception:
                sizes[cid] = 100
        return sizes

    def _init_clients(self) -> None:
        """初始化客户端状态列表。"""
        self._clients = [
            ClientState(client_id=i, data_size=100)
            for i in range(self.config.num_clients)
        ]

    def _update_connectivity(self, timeslot: int) -> None:
        """
        根据通信调度器更新客户端连接状态。

        Parameters
        ----------
        timeslot : int
            当前时间槽。
        """
        if self.scheduler is None:
            # 无调度器：假设始终可通信
            for c in self._clients:
                c.is_connected = True
            return

        connected = set(self.scheduler.get_connected_sats(timeslot))
        for c in self._clients:
            c.is_connected = c.client_id in connected

    def _train_client(
        self,
        client_id: int,
        train_loaders: dict[int, Any],
        round_num: int,
    ) -> ClientUpdate | None:
        """
        训练单个客户端。

        Parameters
        ----------
        client_id : int
            客户端 ID。
        train_loaders : dict[int, DataLoader]
            客户端 ID → DataLoader 映射。
        round_num : int
            当前全局轮次。

        Returns
        -------
        ClientUpdate | None
            训练结果，失败时返回 None。
        """
        if client_id not in train_loaders:
            return None

        try:
            global_weights = [
                param.data.clone()
                for param in self._global_model.parameters()
            ]
            return self.trainer.train(
                client_id=client_id,
                model=self._global_model,
                train_loader=train_loaders[client_id],
                global_weights=global_weights,
                round_num=round_num,
            )
        except Exception as e:
            print(f"  [警告] 客户端 {client_id} 训练失败: {e}")
            return None

    # ── 同步训练模式 (FedAvg / FedProx) ──────────────────────

    def run_sync(
        self,
        model: Any,
        train_loaders: dict[int, Any],
        test_loader: Any,
        verbose: bool = True,
    ) -> list[FLRoundResult]:
        """
        运行同步 FL 训练（适用于 FedAvg / FedProx）。

        通信驱动轮次推进：
            1. 找到当前时刻可通信的卫星 → 分发全局模型
            2. 卫星本地训练（不占虚拟时间）
            3. 等待各卫星下一次接触窗口 → 上传模型
            4. 聚合 → 新全局模型诞生 → 本轮完成
            5. 时间跳到聚合后下一次有卫星连接的时隙 → 下一轮

        Parameters
        ----------
        model : nn.Module
            初始全局模型。
        train_loaders : dict[int, DataLoader]
            客户端 ID → 本地训练数据 Loader。
        test_loader : DataLoader
            测试数据 Loader。
        verbose : bool
            是否打印进度。

        Returns
        -------
        list[FLRoundResult]
            每轮训练结果。
        """
        self._global_model = copy.deepcopy(model)
        self._init_clients()
        self._history = []

        # 获取模拟器引用（用于通信窗口查询）
        sim = self.scheduler._sim if self.scheduler is not None else None
        max_ts = sim.num_timeslots if sim else 999999

        # 计算模型大小（字节），供时间模型使用
        model_size_bytes = sum(
            p.numel() * p.element_size()
            for p in self._global_model.parameters()
        )

        # 预计算客户端数据量
        client_data_sizes = self._get_client_data_sizes(train_loaders)

        current_ts = 0  # 当前虚拟时间槽
        completed_rounds = 0  # 已完成轮次计数

        while completed_rounds < self.config.num_rounds:
            # ── 时间分解记录 ──
            breakdown = TimeBreakdown()

            # ── 1. 找到下一个有卫星连接的时隙（分发窗口）──
            start_ts = self._advance_to_next_contact(current_ts, max_ts)
            if start_ts is None:
                if verbose:
                    print(f"  轮次 {completed_rounds + 1}: 无更多通信窗口，训练终止")
                break

            breakdown.wait_distribution = start_ts - current_ts
            current_ts = start_ts

            # ── 1b. 模型下载耗时 ──
            download_slots = self.time_model.compute_download_slots(model_size_bytes)
            breakdown.download = download_slots
            current_ts += download_slots

            # ── 2. 更新通信状态，选择客户端 ──
            self._update_connectivity(current_ts)
            selected_ids = self.selector.select(self._clients, completed_rounds)
            if len(selected_ids) < 1:
                # 无可选客户端，前进1个时隙重试
                current_ts += 1
                continue

            # ── 3. 本地训练（时间模型决定耗时）──
            train_start_ts = current_ts
            updates: list[ClientUpdate] = []
            max_train_slots = 0
            for cid in selected_ids:
                update = self._train_client(cid, train_loaders, completed_rounds)
                if update is not None:
                    updates.append(update)
                    # 计算该客户端训练耗时
                    n_samples = client_data_sizes.get(cid, 100)
                    train_slots = self.time_model.compute_train_slots(
                        cid, n_samples, self.config.local_epochs,
                    )
                    breakdown.per_satellite[cid] = {"train": train_slots}
                    max_train_slots = max(max_train_slots, train_slots)

            if not updates:
                current_ts += 1
                continue

            breakdown.train = max_train_slots
            current_ts = train_start_ts + max_train_slots

            # ── 4. 等待各卫星返回（下一次接触窗口 + 上传时间）──
            if sim is not None:
                return_times = []
                return_start_ts = current_ts
                for cid in selected_ids:
                    next_contact = sim.get_next_contact(cid, current_ts)
                    if next_contact is not None:
                        contact_ts = next_contact[0]
                        # 上传时间叠加在接触窗口之后
                        upload_slots = self.time_model.compute_upload_slots(
                            model_size_bytes,
                        )
                        breakdown.per_satellite.setdefault(cid, {})
                        breakdown.per_satellite[cid]["upload"] = upload_slots
                        breakdown.per_satellite[cid]["wait_return"] = (
                            contact_ts - current_ts
                        )
                        return_times.append(contact_ts + upload_slots)
                if return_times:
                    current_ts = max(return_times)
                    breakdown.wait_return = current_ts - return_start_ts
                # 如果无返回窗口，模型更新丢失，但仍聚合已返回的
            else:
                # 无模拟器：固定步进
                current_ts += self.config.timeslots_per_round
                breakdown.wait_return = self.config.timeslots_per_round

            # 上传时间从 per_satellite 汇总
            breakdown.upload = max(
                (v.get("upload", 0) for v in breakdown.per_satellite.values()),
                default=0,
            )
            # 去掉上传时间重复计算（wait_return 已包含 upload 叠加）
            # 这里 wait_return 记录的是从 train_end 到 return_ts 的总等待时间
            # upload 单独记录以便分解展示

            # ── 5. 聚合 → 新全局模型诞生 ──
            global_weights = [
                param.data.clone()
                for param in self._global_model.parameters()
            ]
            if self.aggregator.should_aggregate(updates, completed_rounds):
                new_weights = self.aggregator.aggregate(
                    global_weights, updates, completed_rounds,
                )
                with torch.no_grad():
                    for param, new_w in zip(
                        self._global_model.parameters(), new_weights
                    ):
                        param.data.copy_(new_w)

            # 标记客户端参与
            for cid in selected_ids:
                client = self._clients[cid]
                client.last_update_round = completed_rounds

            # ── 6. 评估 ──
            eval_metrics = self.evaluator.evaluate(
                self._global_model, test_loader, completed_rounds,
            )

            # 计算时间分解总计
            breakdown.total = current_ts - start_ts

            avg_loss = sum(u.train_loss for u in updates) / len(updates)
            result = FLRoundResult(
                round_num=completed_rounds,
                num_clients=len(updates),
                train_loss=round(avg_loss, 6),
                eval_metrics=eval_metrics,
                timeslot=current_ts,
                timeslot_start=start_ts,
                time_breakdown=breakdown.to_dict(),
            )
            self._history.append(result)

            completed_rounds += 1

            # 准备下一轮：从聚合时刻之后开始
            current_ts += 1

            if verbose:
                acc = eval_metrics.get("accuracy", 0)
                conn_at_start = (
                    len(self.scheduler.get_connected_sats(start_ts))
                    if self.scheduler else len(selected_ids)
                )
                # 构建时间分解显示
                tm_display = self.time_model.slots_to_display(
                    breakdown.total,
                    getattr(self.time_model, "timeslot_duration_min", 1.0),
                )
                parts = []
                if breakdown.download > 0:
                    parts.append(f"下载:{breakdown.download}")
                if breakdown.train > 0:
                    parts.append(f"训练:{breakdown.train}")
                if breakdown.upload > 0:
                    parts.append(f"上传:{breakdown.upload}")
                time_info = " | ".join(parts) if parts else "瞬时"
                print(
                    f"  轮次 {completed_rounds:3d}/{self.config.num_rounds} | "
                    f"TS={start_ts:4d}→{result.timeslot:4d} "
                    f"({tm_display}) | "
                    f"在线:{conn_at_start} | "
                    f"选中:{len(updates):2d} | "
                    f"耗时: {time_info} | "
                    f"准确率:{acc:.4f}"
                )

        return self._history

    def _advance_to_next_contact(
        self, from_ts: int, max_ts: int
    ) -> int | None:
        """
        从 from_ts 开始找到下一个有任意卫星连接的时隙。

        如果没有调度器，直接返回 from_ts（假设始终可通信）。
        如果超过 max_ts 仍无连接，返回 None。
        """
        if self.scheduler is None:
            return from_ts

        for ts in range(from_ts, max_ts):
            connected = self.scheduler.get_connected_sats(ts)
            if connected:
                return ts
        return None

    # ── 异步训练模式 (FedBuff) ──────────────────────────────

    def run_async(
        self,
        model: Any,
        train_loaders: dict[int, Any],
        test_loader: Any,
        verbose: bool = True,
    ) -> list[FLRoundResult]:
        """
        运行异步 FL 训练（适用于 FedBuff）。

        流程：
            与同步不同，客户端独立训练并随时提交更新。
            服务端在每个 timeslot：
                1. 更新通信状态
                2. 让所有可通信的客户端分别训练
                3. 将更新放入缓冲区
                4. 缓冲区满时触发聚合
                5. 周期性评估

        Parameters
        ----------
        model : nn.Module
            初始全局模型。
        train_loaders : dict[int, DataLoader]
            客户端 ID → 本地训练数据 Loader。
        test_loader : DataLoader
            测试数据 Loader。
        verbose : bool
            是否打印进度。

        Returns
        -------
        list[FLRoundResult]
            聚合事件记录。
        """
        from fl_space.fl.fedbuff import BufferAggregator

        self._global_model = copy.deepcopy(model)
        self._init_clients()
        self._history = []

        if not isinstance(self.aggregator, BufferAggregator):
            raise TypeError(
                "异步模式需要 BufferAggregator，当前聚合器: "
                f"{type(self.aggregator).__name__}"
            )

        buffer_agg: BufferAggregator = self.aggregator
        total_timeslots = self.config.num_rounds * self.config.timeslots_per_round
        training_clients: set[int] = set()
        global_round = 0

        eval_interval = max(1, total_timeslots // 20)  # 约 20 次评估
        last_eval_metrics: dict[str, float] = {}

        for ts in range(total_timeslots):
            # 1. 更新通信状态
            self._update_connectivity(ts)

            # 2. 选择可参与训练的客户端
            available = self.selector.select(
                self._clients, global_round,
                already_training=training_clients,
            )

            # 3. 每个可用客户端独立训练
            for cid in available:
                update = self._train_client(cid, train_loaders, global_round)
                if update is not None:
                    buffer_agg.add_update(update)
                    self._clients[cid].last_update_round = global_round

            # 4. 检查缓冲区是否触发聚合
            if buffer_agg.should_aggregate([], global_round):
                global_weights = [
                    param.data.clone()
                    for param in self._global_model.parameters()
                ]
                new_weights = buffer_agg.aggregate(
                    global_weights, [], global_round,
                )
                with torch.no_grad():
                    for param, new_w in zip(
                        self._global_model.parameters(), new_weights
                    ):
                        param.data.copy_(new_w)

                global_round += 1

                # 5. 周期性评估
                if ts % eval_interval == 0:
                    last_eval_metrics = self.evaluator.evaluate(
                        self._global_model, test_loader, global_round,
                    )

                status = buffer_agg.buffer_status()
                result = FLRoundResult(
                    round_num=global_round,
                    num_clients=status["current_count"],
                    train_loss=0.0,  # 异步模式下难以精确计算
                    eval_metrics=dict(last_eval_metrics),
                    timeslot=ts,
                )
                self._history.append(result)

                if verbose:
                    acc = last_eval_metrics.get("accuracy", 0)
                    print(
                        f"  Timeslot {ts:4d} | 聚合 #{global_round:3d} | "
                        f"缓冲区: {status['current_count']}/{status['buffer_size']} | "
                        f"准确率: {acc:.4f}"
                    )

        # 最终评估
        final_metrics = self.evaluator.evaluate(
            self._global_model, test_loader, global_round,
        )
        if verbose:
            acc = final_metrics.get("accuracy", 0)
            print(f"\n  最终准确率: {acc:.4f}")
            print(f"  总聚合次数: {global_round}")

        return self._history

    # ── 通用接口 ──────────────────────────────────────────────

    def run(
        self,
        model: Any,
        train_loaders: dict[int, Any],
        test_loader: Any,
        verbose: bool = True,
    ) -> list[FLRoundResult]:
        """
        根据配置自动选择同步或异步模式运行。

        Parameters
        ----------
        model : nn.Module
            初始全局模型。
        train_loaders : dict[int, DataLoader]
            客户端 ID → 本地训练数据 Loader。
        test_loader : DataLoader
            测试数据 Loader。
        verbose : bool
            是否打印进度。

        Returns
        -------
        list[FLRoundResult]
            训练历史。

        Raises
        ------
        ValueError
            当 algorithm 未知时。
        """
        algo = self.config.algorithm.lower()
        if algo in ("fedavg", "fedprox"):
            return self.run_sync(model, train_loaders, test_loader, verbose)
        elif algo == "fedbuff":
            return self.run_async(model, train_loaders, test_loader, verbose)
        else:
            raise ValueError(
                f"未知算法: '{algo}'，支持: fedavg, fedprox, fedbuff"
            )

    def get_global_model(self) -> Any:
        """返回当前全局模型。"""
        return self._global_model

    def get_history_dict(self) -> list[dict[str, Any]]:
        """
        将训练历史导出为字典列表（便于 JSON 序列化）。

        Returns
        -------
        list[dict]
            每轮结果字典。
        """
        results = []
        for r in self._history:
            entry: dict[str, Any] = {
                "round": r.round_num,
                "timeslot": r.timeslot,
                "timeslot_start": r.timeslot_start,
                "num_clients": r.num_clients,
                "train_loss": r.train_loss,
            }
            entry.update(r.eval_metrics)
            if r.time_breakdown:
                entry["time_breakdown"] = r.time_breakdown
            results.append(entry)
        return results
