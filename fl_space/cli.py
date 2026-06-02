"""
fl-space CLI — 命令行调参与实验统筹工具

提供统一的命令行入口，覆盖：
    - simulate  : 轨道接触模拟
    - train     : FL 训练实验
    - list      : 查看内置预设、模型、卫星类型
    - export    : 导出模拟结果 JSON
    - info      : 系统/环境信息

所有子命令均支持 --config 参数加载 JSON 配置文件，
CLI 参数将覆盖 JSON 中的对应字段。

设计原则：
    - 零额外依赖：仅使用 argparse（标准库）
    - 每个子命令对应一个独立功能模块
    - 输出统一使用 JSON 行或表格格式，便于脚本调用
    - 遵循 CODING_STANDARDS.md

使用示例::

    fl-space simulate --sats 10 --stations 5 --hours 24
    fl-space simulate --config my_sim.json --hours 48    # JSON + CLI 覆盖
    fl-space train --algo fedprox --dataset cifar10 --rounds 100
    fl-space train --config my_fl.json --lr 0.001        # JSON + CLI 覆盖
    fl-space list presets
    fl-space list models
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from typing import Any


def _check_torch() -> bool:
    """检查 PyTorch 是否可用。"""
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _check_skyfield() -> bool:
    """检查 Skyfield 是否可用。"""
    try:
        import skyfield  # noqa: F401
        return True
    except ImportError:
        return False


# ── 工具函数 ─────────────────────────────────────────────────


def _args_to_dict(args: argparse.Namespace, *keys: str) -> dict:
    """将指定的 argparse 参数提取为字典（仅限非 None 值）。"""
    return {k: v for k in keys if (v := getattr(args, k, None)) is not None}


def _merge_config(
    json_path: str | None,
    cli_args: dict,
    defaults: dict,
) -> dict:
    """
    合并 JSON 配置和 CLI 参数。

    优先级：CLI 显式参数 > JSON 配置 > 默认值

    Parameters
    ----------
    json_path : str | None
        JSON 配置文件路径，None 则不加载。
    cli_args : dict
        CLI 传入的参数。
    defaults : dict
        默认值字典。

    Returns
    -------
    dict
        合并后的配置字典。
    """
    config = dict(defaults)

    # 先加载 JSON 配置
    if json_path:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
            config.update(json_data)
            print(f"  [配置] 加载 {json_path}")
        except FileNotFoundError:
            print(f"  [警告] 配置文件不存在: {json_path}，使用默认值")
        except json.JSONDecodeError as e:
            print(f"  [警告] JSON 解析失败: {e}，使用默认值")

    # CLI 参数覆盖
    config.update({k: v for k, v in cli_args.items() if v is not None})

    return config


# ── 子命令：simulate ─────────────────────────────────────────


def _cmd_simulate(args: argparse.Namespace) -> int:
    """
    运行轨道接触模拟。

    用法: fl-space simulate [参数]

    支持 --config 加载 JSON 配置文件，CLI 参数覆盖 JSON 字段。
    JSON 配置格式见 CLI_REFERENCE.md。

    示例:
        fl-space simulate
        fl-space simulate --sats 20 --stations 10 --hours 48
        fl-space simulate --backend skyfield --altitude 550 --inclination 53
        fl-space simulate --config my_sim.json --hours 48
        fl-space simulate --sats 10 --stations 5 --output result.json
    """
    import json as _json

    from fl_space.environment import CelestialBody, create_default_network
    from fl_space.orbit import create_circular_orbit
    from fl_space.simulator import OrbitSimulator

    num_slots = int(args.hours * 60 / args.timeslot_duration)

    # 合并配置：默认 → JSON → CLI
    cfg = _merge_config(
        json_path=args.config,
        cli_args={
            "num_satellites": args.sats,
            "num_ground_stations": args.stations,
            "orbit_altitude_km": args.altitude,
            "orbit_inclination_deg": args.inclination,
            "distribution": args.distribution,
            "backend": args.backend,
            "num_timeslots": num_slots,
            "timeslot_duration_min": args.timeslot_duration,
        },
        defaults={
            "num_satellites": 5,
            "num_ground_stations": 3,
            "orbit_altitude_km": 500.0,
            "orbit_inclination_deg": 90.0,
            "distribution": "uniform",
            "backend": "kepler",
            "num_timeslots": num_slots,
            "timeslot_duration_min": 1.0,
        },
    )

    n_sats = int(cfg["num_satellites"])
    n_gs = int(cfg["num_ground_stations"])

    # 天体
    if "body" in cfg and isinstance(cfg["body"], dict):
        body = CelestialBody.from_dict(cfg["body"])
    else:
        body = CelestialBody.earth()

    # 地面站
    gs_network = create_default_network(n_gs)

    # 轨道
    orbits = []
    for i in range(n_sats):
        raan = (360.0 / n_sats) * i if cfg["distribution"] == "uniform" else i * 72.0
        orb = create_circular_orbit(
            altitude_km=cfg["orbit_altitude_km"],
            inclination_deg=cfg["orbit_inclination_deg"],
            raan_deg=raan,
            true_anomaly_deg=i * (360.0 / n_sats),
            body=body,
        )
        orbits.append(orb)

    print(f"=== 轨道接触模拟 ===")
    print(f"  卫星数: {n_sats}")
    print(f"  地面站: {n_gs}")
    print(f"  时长: {args.hours} 小时")
    print(f"  后端: {cfg['backend']}")
    print(f"  轨道高度: {cfg['orbit_altitude_km']} km")
    print(f"  轨道倾角: {cfg['orbit_inclination_deg']}°")
    print(f"  分布策略: {cfg['distribution']}")
    print()

    sim = OrbitSimulator(
        body=body,
        orbits=orbits,
        ground_station_network=gs_network,
        num_timeslots=cfg["num_timeslots"],
        timeslot_duration_min=cfg["timeslot_duration_min"],
        backend=cfg["backend"],
        distribution=cfg["distribution"],
        verbose=not args.quiet,
    )

    # 输出摘要
    print()
    print(f"=== 结果摘要 ===")
    contact_rate = sim.stats.get("contact_rate", 0)
    print(f"  接触率: {contact_rate:.2%}")
    print(f"  总接触数: {sim.stats.get('total_contacts', 'N/A')}")
    print(f"  平均每星接触: {sim.stats.get('avg_contacts_per_sat', 'N/A')}")

    # 输出通信记录
    if args.show_contacts:
        print()
        print("--- 通信记录摘要 ---")
        for sat_id in range(args.sats):
            record = sim.get_communication_record(sat_id)
            windows = len(set(
                e.get("timeslot", 0) for e in record if e.get("in_contact")
            )) if record else 0
            print(f"  SAT-{sat_id}: {windows} 个接触时段")

    # 导出 JSON
    if args.output:
        output_data = {
            "config": {
                "num_satellites": args.sats,
                "num_ground_stations": args.stations,
                "duration_hours": args.hours,
                "altitude_km": args.altitude,
                "inclination_deg": args.inclination,
                "backend": args.backend,
                "distribution": args.distribution,
            },
            "stats": sim.stats,
            "contact_rate": sim.stats.get("contact_rate", 0),
            "communication_records": {
                sat_id: sim.get_communication_record(sat_id)
                for sat_id in range(args.sats)
            },
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n  结果已导出至: {args.output}")

    return 0


# ── 子命令：train ────────────────────────────────────────────


def _cmd_train(args: argparse.Namespace) -> int:
    """
    运行 FL 训练实验。

    用法: fl-space train [参数]

    支持 --config 加载 JSON 配置文件，CLI 参数覆盖 JSON 字段。
    JSON 配置格式见 CLI_REFERENCE.md。

    示例:
        fl-space train
        fl-space train --algo fedprox --dataset cifar10 --rounds 100
        fl-space train --algo fedavg --scale medium --epochs 10 --lr 0.01
        fl-space train --config my_fl.json --lr 0.001
        fl-space train --algo fedbuff --buffer-size 10 --non-iid
    """
    if not _check_torch():
        print("错误: FL 训练需要 PyTorch。请运行: pip install fl-space[full]")
        return 1

    from fl_space.fl.runner import FLRunner
    from fl_space.fl.server import FLConfig

    # ── 如果提供 --config，从 JSON 加载基础配置 ──
    if args.config:
        base_config = FLConfig.from_json(args.config)
        algo = base_config.algorithm
        device = base_config.device
        print(f"  [配置] 加载 {args.config} (算法: {algo})")
    else:
        algo = args.algo
        device = args.device

    # 组装 CLI 覆盖参数
    overrides: dict[str, Any] = {}
    if args.rounds is not None:
        overrides["num_rounds"] = args.rounds
    if args.epochs != 5:  # 与 argparse 默认值比较
        overrides["local_epochs"] = args.epochs
    if args.lr != 0.01:
        overrides["learning_rate"] = args.lr
    if args.batch_size != 32:
        overrides["batch_size"] = args.batch_size
    if args.device != "cpu":
        overrides["device"] = args.device
    if args.seed is not None:
        overrides["seed"] = args.seed
    if algo == "fedprox" and args.mu != 0.01:
        overrides["mu"] = args.mu
    if algo == "fedbuff":
        if args.buffer_size != 5:
            overrides["buffer_size"] = args.buffer_size
        if args.staleness:
            overrides["staleness_weight"] = True

    # 时间模型参数
    if args.time_model is not None:
        overrides["time_model"] = args.time_model
    if args.time_model_args is not None:
        try:
            overrides["time_model_kwargs"] = json.loads(args.time_model_args)
        except json.JSONDecodeError as e:
            print(f"错误: --time-model-args JSON 解析失败: {e}")
            return 1

    # 如果有 --config，用自定义组件构建 Runner
    if args.config:
        from fl_space.fl.fedavg import create_fedavg_components
        from fl_space.fl.fedprox import create_fedprox_components
        from fl_space.fl.fedbuff import create_fedbuff_components

        # 将 CLI 覆盖合并到 base_config
        for k, v in overrides.items():
            if hasattr(base_config, k):
                setattr(base_config, k, v)
        base_config.algorithm = algo
        base_config.device = device

        component_factories = {
            "fedavg": create_fedavg_components,
            "fedprox": create_fedprox_components,
            "fedbuff": create_fedbuff_components,
        }
        factory = component_factories.get(algo.lower())
        if factory is None:
            print(f"错误: 未知算法 '{algo}'")
            return 1

        components = factory(
            fraction=base_config.fraction,
            min_clients=max(1, int(base_config.num_clients * base_config.fraction)),
            local_epochs=base_config.local_epochs,
            batch_size=base_config.batch_size,
            learning_rate=base_config.learning_rate,
            device=base_config.device,
            seed=base_config.seed,
            **({"mu": base_config.mu} if algo == "fedprox" else {}),
            **({
                "buffer_size": base_config.buffer_size,
                "staleness_weight": base_config.staleness_weight,
            } if algo == "fedbuff" else {}),
        )

        runner = FLRunner(base_config, *components)
    else:
        runner = FLRunner.from_preset(
            algorithm=algo,
            scale=args.scale,
            dataset=args.dataset,
            device=device,
            **overrides,
        )

    print(f"=== FL 训练实验 ===")
    print(f"  算法: {algo}")
    print(f"  数据集: {args.dataset}")
    print(f"  规模: {args.scale}")
    if hasattr(runner, "config"):
        print(f"  轮次: {runner.config.num_rounds}")
        print(f"  本地 epoch: {runner.config.local_epochs}")
        print(f"  学习率: {runner.config.learning_rate}")
        print(f"  batch size: {runner.config.batch_size}")
        print(f"  时间模型: {runner.config.time_model}")
        if runner.config.time_model_kwargs:
            print(f"  时间模型参数: {runner.config.time_model_kwargs}")
    print(f"  设备: {device}")
    if algo == "fedprox":
        mu_val = runner.config.mu if hasattr(runner, "config") else args.mu
        print(f"  近端项 μ: {mu_val}")
    if algo == "fedbuff":
        bs = runner.config.buffer_size if hasattr(runner, "config") else args.buffer_size
        print(f"  缓冲区 K: {bs}")
        print(f"  staleness降权: {args.staleness}")
    print()

    history = runner.run(
        dataset_name=args.dataset,
        iid=not args.non_iid,
        alpha=args.alpha,
        verbose=not args.quiet,
    )

    # 导出 JSON
    if args.output:
        output_data = {
            "config": {
                "algorithm": algo,
                "dataset": args.dataset,
                "scale": args.scale,
                "rounds": args.rounds,
                "local_epochs": args.epochs,
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "iid": not args.non_iid,
                "alpha": args.alpha,
            },
            "history": runner.history_dict,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n  结果已导出至: {args.output}")

    return 0


# ── 子命令：list ─────────────────────────────────────────────


def _cmd_list(args: argparse.Namespace) -> int:
    """
    列出可用资源。

    用法: fl-space list <资源类型>

    示例:
        fl-space list presets
        fl-space list models
        fl-space list satellites
        fl-space list experiments
    """
    resource = args.resource

    if resource == "presets":
        from fl_space.fl.config import EXPERIMENT_PRESETS
        print("=== FL 实验预设 ===\n")
        for p in EXPERIMENT_PRESETS:
            print(f"  {p['name']:30s} — {p['description']}")

    elif resource == "models":
        try:
            from fl_space.fl.models import list_models
            print("=== 可用模型 ===\n")
            models = list_models()
            if models:
                for m in models:
                    print(f"  - {m}")
            else:
                print("  (PyTorch 未安装，无可用模型)")
        except ImportError:
            print("  (PyTorch 未安装，无可用模型)")

    elif resource == "satellites":
        from fl_space.orbit.satellite_registry import registry
        print("=== 已注册卫星类型 ===\n")
        types = registry.list_types()
        if types:
            for t in types:
                if isinstance(t, dict):
                    name = t.get("name", "?")
                    desc = t.get("description", "")
                    cat = t.get("category", "")
                    print(f"  {name:20s} [{cat}] — {desc}")
                else:
                    print(f"  - {t}")
        else:
            print("  (无已注册卫星类型)")

    elif resource == "experiments":
        from fl_space.config.defaults import EXPERIMENT_CONFIGS
        print("=== 模拟实验预设 ===\n")
        for cfg in EXPERIMENT_CONFIGS:
            name, sats, gss, desc = cfg
            print(f"  {name:15s} — {sats} 卫星, {gss} 地面站 — {desc}")

    else:
        print(f"未知资源类型: {resource}")
        print("可用类型: presets, models, satellites, experiments")
        return 1

    return 0


# ── 子命令：export ───────────────────────────────────────────


def _cmd_export(args: argparse.Namespace) -> int:
    """
    将模拟结果导出为 JSON。

    用法: fl-space export [参数]

    示例:
        fl-space export --sats 10 --stations 5 --output sim_result.json
        fl-space export --body mars --sats 5 --output mars_sim.json
    """
    from fl_space.environment import CelestialBody, create_default_network
    from fl_space.orbit import create_circular_orbit
    from fl_space.simulator import OrbitSimulator

    # 选择天体
    body_map = {
        "earth": CelestialBody.earth,
        "mars": CelestialBody.mars,
        "moon": CelestialBody.moon,
        "jupiter": CelestialBody.jupiter,
        "saturn": CelestialBody.saturn,
        "venus": CelestialBody.venus,
    }
    if args.body not in body_map:
        print(f"未知天体: {args.body}，可用: {list(body_map.keys())}")
        return 1

    body = body_map[args.body]()
    gs_network = create_default_network(args.stations)

    orbits = []
    for i in range(args.sats):
        orb = create_circular_orbit(
            altitude_km=args.altitude,
            inclination_deg=args.inclination,
            raan_deg=i * (360.0 / args.sats),
            true_anomaly_deg=i * (360.0 / args.sats),
            body=body,
        )
        orbits.append(orb)

    num_slots = int(args.hours * 60 / args.timeslot)
    sim = OrbitSimulator(
        body=body,
        orbits=orbits,
        ground_station_network=gs_network,
        num_timeslots=num_slots,
        timeslot_duration_min=args.timeslot,
        backend=args.backend,
        verbose=False,
    )

    export_data = {
        "config": {
            "body": args.body,
            "num_satellites": args.sats,
            "num_ground_stations": args.stations,
            "duration_hours": args.hours,
            "altitude_km": args.altitude,
            "inclination_deg": args.inclination,
            "backend": args.backend,
        },
        "contact_rate": sim.stats.get("contact_rate", 0),
        "stats": sim.stats,
        "communication_records": {
            f"SAT-{sat_id}": sim.get_communication_record(sat_id)
            for sat_id in range(args.sats)
        },
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)

    print(f"导出完成: {args.output}")
    print(f"  天体: {args.body}")
    print(f"  卫星数: {args.sats}")
    print(f"  接触率: {sim.stats.get('contact_rate', 0):.2%}")

    return 0


# ── 子命令：info ─────────────────────────────────────────────


def _cmd_info(args: argparse.Namespace) -> int:  # noqa: ARG001
    """
    显示系统与环境信息。

    用法: fl-space info
    """
    import platform

    from fl_space import __version__

    # 使用纯 ASCII 标记以避免 Windows GBK 编码问题
    ok = "[OK]"
    no = "[--]"

    print("=== SpaceFL 环境信息 ===\n")
    print(f"  框架版本: {__version__}")
    print(f"  Python:   {platform.python_version()} ({platform.python_implementation()})")
    print(f"  操作系统: {platform.system()} {platform.release()}")
    print(f"  PyTorch:  {ok + ' 可用' if _check_torch() else no + ' 未安装'}")
    print(f"  Skyfield: {ok + ' 可用' if _check_skyfield() else no + ' 未安装'}")
    print(f"  NumPy:    ", end="")
    try:
        import numpy
        print(f"{ok} {numpy.__version__}")
    except ImportError:
        print(f"{no} 未安装")
    print(f"  Matplotlib: ", end="")
    try:
        import matplotlib
        print(f"{ok} {matplotlib.__version__}")
    except ImportError:
        print(f"{no} 未安装")
    print(f"  CUDA:     ", end="")
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        print(f"{ok + ' ' + torch.version.cuda if cuda_ok else no + ' 不可用 (CPU only)'}")
    except ImportError:
        print(f"{no} PyTorch 未安装")

    return 0


# ── 主入口 ───────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """
    构建完整的 CLI 参数解析器。

    Returns
    -------
    argparse.ArgumentParser
        配置好的参数解析器。
    """
    parser = argparse.ArgumentParser(
        prog="fl-space",
        description="SpaceFL — 太空联邦学习研究框架命令行工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  fl-space simulate --sats 10 --stations 5 --hours 24
  fl-space train --algo fedprox --dataset cifar10 --rounds 100
  fl-space list presets
  fl-space export --body mars --sats 5 --output mars.json
  fl-space info
        """,
    )

    sub = parser.add_subparsers(dest="command", title="子命令")

    # ── simulate ──────────────────────────────────────────

    p_sim = sub.add_parser("simulate", help="运行轨道接触模拟")
    p_sim.add_argument("--config", "-c", type=str, default=None,
                       help="JSON 配置文件路径（CLI 参数将覆盖 JSON 中字段）")
    p_sim.add_argument("--sats", "-n", type=int, default=5, help="卫星数量 (默认: 5)")
    p_sim.add_argument("--stations", "-g", type=int, default=3, help="地面站数量 (默认: 3)")
    p_sim.add_argument("--hours", "-t", type=float, default=24, help="模拟时长/小时 (默认: 24)")
    p_sim.add_argument("--backend", "-b", choices=["kepler", "skyfield"], default="kepler",
                       help="后端引擎 (默认: kepler)")
    p_sim.add_argument("--altitude", "-a", type=float, default=500.0,
                       help="轨道高度 km (默认: 500)")
    p_sim.add_argument("--inclination", "-i", type=float, default=90.0,
                       help="轨道倾角 ° (默认: 90)")
    p_sim.add_argument("--distribution", "-d", choices=["walker", "cluster", "uniform"],
                       default="uniform", help="星座分布策略 (默认: uniform)")
    p_sim.add_argument("--timeslot-duration", type=float, default=1.0,
                       help="每时间槽分钟数 (默认: 1.0)")
    p_sim.add_argument("--output", "-o", type=str, default=None,
                       help="导出 JSON 文件路径")
    p_sim.add_argument("--generate-config", type=str, default=None, metavar="FILE",
                       help="生成模拟器配置模板 JSON 到指定文件")
    p_sim.add_argument("--show-contacts", action="store_true",
                       help="显示通信记录摘要")
    p_sim.add_argument("--quiet", "-q", action="store_true",
                       help="安静模式，减少输出")
    p_sim.set_defaults(func=_cmd_simulate)

    # ── train ─────────────────────────────────────────────

    p_train = sub.add_parser("train", help="运行 FL 训练实验")
    p_train.add_argument("--config", "-c", type=str, default=None,
                         help="JSON 配置文件路径（CLI 参数将覆盖 JSON 中字段）")
    p_train.add_argument("--algo", choices=["fedavg", "fedprox", "fedbuff"],
                         default="fedavg", help="FL 算法 (默认: fedavg)")
    p_train.add_argument("--dataset", "-d", choices=["mnist", "fashion_mnist", "cifar10"],
                         default="mnist", help="数据集 (默认: mnist)")
    p_train.add_argument("--scale", "-s", choices=["small", "medium", "large"],
                         default="small", help="实验规模 (默认: small)")
    p_train.add_argument("--rounds", "-r", type=int, default=None,
                         help="全局训练轮次 (覆盖规模默认值)")
    p_train.add_argument("--epochs", "-e", type=int, default=5,
                         help="本地训练 epoch 数 (默认: 5)")
    p_train.add_argument("--lr", type=float, default=0.01,
                         help="学习率 (默认: 0.01)")
    p_train.add_argument("--batch-size", type=int, default=32,
                         help="batch size (默认: 32)")
    p_train.add_argument("--mu", type=float, default=0.01,
                         help="FedProx 近端项系数 μ (默认: 0.01)")
    p_train.add_argument("--buffer-size", type=int, default=5,
                         help="FedBuff 缓冲区大小 K (默认: 5)")
    p_train.add_argument("--staleness", action="store_true",
                         help="FedBuff 启用陈旧度降权")
    p_train.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                         help="计算设备 (默认: cpu)")
    p_train.add_argument("--seed", type=int, default=None,
                         help="随机种子")
    p_train.add_argument("--time-model", type=str, default=None, metavar="MODEL",
                         help="虚拟时间模型: slot|physics|path/to/file.py:ClassName (默认: slot)")
    p_train.add_argument("--time-model-args", type=str, default=None, metavar="JSON",
                         help="时间模型参数 (JSON字符串), 如 '{\"slots_per_epoch\":2}'")
    p_train.add_argument("--non-iid", action="store_true",
                         help="使用 non-IID 数据分布 (默认 IID)")
    p_train.add_argument("--alpha", type=float, default=0.5,
                         help="non-IID Dirichlet alpha (默认: 0.5)")
    p_train.add_argument("--output", "-o", type=str, default=None,
                         help="导出训练历史 JSON 文件路径")
    p_train.add_argument("--generate-config", type=str, default=None, metavar="FILE",
                         help="生成 FL 实验配置模板 JSON 到指定文件")
    p_train.add_argument("--quiet", "-q", action="store_true",
                         help="安静模式，减少输出")
    p_train.set_defaults(func=_cmd_train)

    # ── list ──────────────────────────────────────────────

    p_list = sub.add_parser("list", help="列出可用资源")
    p_list.add_argument("resource", nargs="?", default="presets",
                        choices=["presets", "models", "satellites", "experiments"],
                        help="资源类型: presets|models|satellites|experiments")
    p_list.set_defaults(func=_cmd_list)

    # ── export ────────────────────────────────────────────

    p_export = sub.add_parser("export", help="导出模拟结果为 JSON")
    p_export.add_argument("--output", "-o", type=str, required=True,
                          help="输出 JSON 文件路径")
    p_export.add_argument("--body", choices=["earth", "mars", "moon", "jupiter", "saturn", "venus"],
                          default="earth", help="中心天体 (默认: earth)")
    p_export.add_argument("--sats", "-n", type=int, default=10, help="卫星数量 (默认: 10)")
    p_export.add_argument("--stations", "-g", type=int, default=5, help="地面站数量 (默认: 5)")
    p_export.add_argument("--hours", "-t", type=float, default=24, help="模拟时长/小时 (默认: 24)")
    p_export.add_argument("--altitude", "-a", type=float, default=500.0,
                           help="轨道高度 km (默认: 500)")
    p_export.add_argument("--inclination", "-i", type=float, default=90.0,
                           help="轨道倾角 ° (默认: 90)")
    p_export.add_argument("--backend", "-b", choices=["kepler", "skyfield"], default="kepler",
                           help="后端引擎 (默认: kepler)")
    p_export.add_argument("--timeslot", type=float, default=1.0,
                           help="每时间槽分钟数 (默认: 1.0)")
    p_export.set_defaults(func=_cmd_export)

    # ── info ──────────────────────────────────────────────

    p_info = sub.add_parser("info", help="显示系统与环境信息")
    p_info.set_defaults(func=_cmd_info)

    return parser


# ── 配置模板生成 ─────────────────────────────────────────────


def _cmd_generate_config(args: argparse.Namespace, filepath: str) -> int:
    """
    生成配置模板 JSON 文件。

    根据子命令类型生成对应的模板：
        simulate → 模拟器配置模板
        train    → FL 实验配置模板
    """
    cmd = getattr(args, "command", "simulate")

    if cmd == "simulate":
        template = {
            "_comment": "SpaceFL 模拟器配置模板 — 修改后使用 fl-space simulate --config THIS_FILE.json",
            "num_satellites": 10,
            "num_ground_stations": 5,
            "orbit_altitude_km": 500.0,
            "orbit_inclination_deg": 90.0,
            "distribution": "uniform",
            "backend": "kepler",
            "timeslot_duration_min": 1.0,
            "num_timeslots": 1440,
            "body": {
                "name": "Earth",
                "radius_km": 6371.0,
                "GM": 398600.4418,
                "rotation_period_hours": 24.0,
                "atmosphere_height_km": 100.0,
            },
            "ground_stations": [
                {"name": "Beijing", "lat_deg": 39.9, "lon_deg": 116.4, "altitude_km": 0.05},
                {"name": "Sanya", "lat_deg": 18.25, "lon_deg": 109.5, "altitude_km": 0.0},
            ],
        }
    else:
        template = {
            "_comment": "SpaceFL FL 实验配置模板 — 修改后使用 fl-space train --config THIS_FILE.json",
            "algorithm": "fedavg",
            "num_rounds": 50,
            "num_clients": 10,
            "fraction": 0.5,
            "local_epochs": 5,
            "batch_size": 32,
            "learning_rate": 0.01,
            "mu": 0.01,
            "buffer_size": 5,
            "staleness_weight": False,
            "device": "cpu",
            "seed": 42,
        }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2)

    print(f"配置模板已生成: {filepath}")
    print(f"  编辑此文件后使用: fl-space {cmd} --config {filepath}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """
    CLI 主入口。

    Parameters
    ----------
    argv : list[str] | None
        命令行参数，None 表示使用 sys.argv。

    Returns
    -------
    int
        退出码，0 表示成功。
    """
    parser = build_parser()

    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        parser.print_help()
        return 0

    args = parser.parse_args(argv)

    # ── 处理 --generate-config（生成模板后直接退出） ──
    gen_cfg = getattr(args, "generate_config", None)
    if gen_cfg:
        return _cmd_generate_config(args, gen_cfg)

    if not hasattr(args, "func"):
        parser.print_help()
        return 0

    try:
        return args.func(args)
    except ImportError as e:
        print(f"依赖缺失: {e}")
        print("提示: 请安装所需依赖。例如: pip install fl-space[full]")
        return 1
    except Exception as e:
        print(f"错误: {e}")
        if "--quiet" not in argv:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
