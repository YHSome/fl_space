# fl-space CLI 参考手册

> SpaceFL 命令行工具 — 完整的调参、实验和外接配置指南

---

## 目录

1. [快速开始](#1-快速开始)
2. [子命令速查](#2-子命令速查)
3. [simulate — 轨道接触模拟](#3-simulate--轨道接触模拟)
4. [train — FL 训练实验](#4-train--fl-训练实验)
5. [list — 查看资源](#5-list--查看资源)
6. [export — 导出模拟结果](#6-export--导出模拟结果)
7. [info — 系统环境](#7-info--系统环境)
8. [外接自定义配置](#8-外接自定义配置)
9. [JSON 配置文件参考](#9-json-配置文件参考)

---

## 1. 快速开始

```powershell
# 安装
pip install -e "."

# 验证
fl-space info

# 第一个模拟
fl-space simulate --sats 5 --stations 3 --hours 6
```

---

## 2. 子命令速查

| 命令 | 功能 | 典型场景 |
|------|------|---------|
| `simulate` | 轨道接触模拟 | 探索不同星座配置的接触率 |
| `train` | FL 训练实验 | 对比 FedAvg/FedProx/FedBuff 算法效果 |
| `list` | 查看内置资源 | 查询可用预设、模型、卫星类型 |
| `export` | 导出模拟结果 | 保存接触矩阵到 JSON 供分析 |
| `info` | 系统环境 | 检查依赖安装状态 |

---

## 3. simulate — 轨道接触模拟

### 3.1 基本用法

```powershell
fl-space simulate [参数]
```

### 3.2 完整参数表

| 参数 | 简写 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--sats` | `-n` | int | 5 | 卫星数量 |
| `--stations` | `-g` | int | 3 | 地面站数量 |
| `--hours` | `-t` | float | 24 | 模拟时长（小时） |
| `--backend` | `-b` | kepler \| skyfield | kepler | 轨道后端引擎 |
| `--altitude` | `-a` | float | 500.0 | 轨道高度 (km) |
| `--inclination` | `-i` | float | 90.0 | 轨道倾角 (°) |
| `--distribution` | `-d` | walker \| cluster \| uniform | uniform | 星座分布策略 |
| `--timeslot-duration` | – | float | 1.0 | 每时间槽分钟数 |
| `--config` | `-c` | str | – | JSON 配置文件路径 |
| `--output` | `-o` | str | – | 导出结果 JSON |
| `--show-contacts` | – | flag | – | 显示通信记录摘要 |
| `--generate-config` | – | str | – | 生成配置模板到指定文件 |
| `--quiet` | `-q` | flag | – | 安静模式 |

### 3.3 调参示例

```powershell
# 基础模拟
fl-space simulate

# 大规模星座 + 高精度后端
fl-space simulate --sats 30 --stations 13 --hours 48 --backend skyfield

# 低轨 Walker 星座
fl-space simulate -n 20 -g 10 -a 550 -i 53 -d walker -t 24

# 导出结果 JSON
fl-space simulate -n 10 -g 5 --output sim_result.json

# 从 JSON 配置文件加载 + CLI 覆盖某些参数
fl-space simulate --config my_config.json --hours 72 --quiet
```

---

## 4. train — FL 训练实验

### 4.1 基本用法

```powershell
fl-space train [参数]
```

### 4.2 完整参数表

| 参数 | 简写 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--algo` | – | fedavg \| fedprox \| fedbuff | fedavg | FL 算法 |
| `--dataset` | `-d` | mnist \| fashion_mnist \| cifar10 | mnist | 数据集 |
| `--scale` | `-s` | small \| medium \| large | small | 实验规模 |
| `--rounds` | `-r` | int | 规模决定 | 全局训练轮次 |
| `--epochs` | `-e` | int | 5 | 本地训练 epoch 数 |
| `--lr` | – | float | 0.01 | 学习率 |
| `--batch-size` | – | int | 32 | batch size |
| `--mu` | – | float | 0.01 | FedProx 近端项系数 μ |
| `--buffer-size` | – | int | 5 | FedBuff 缓冲区大小 K |
| `--staleness` | – | flag | – | FedBuff 陈旧度降权 |
| `--device` | – | cpu \| cuda | cpu | 计算设备 |
| `--seed` | – | int | – | 随机种子 |
| `--non-iid` | – | flag | – | 使用 non-IID 数据分布 |
| `--alpha` | – | float | 0.5 | Dirichlet α (non-IID) |
| `--config` | `-c` | str | – | JSON 配置文件路径 |
| `--output` | `-o` | str | – | 导出训练历史 JSON |
| `--generate-config` | – | str | – | 生成配置模板到指定文件 |
| `--quiet` | `-q` | flag | – | 安静模式 |

### 4.3 规模预设对应关系

| 规模 | 客户端数 | 默认轮次 | 适用场景 |
|------|---------|---------|---------|
| `small` | 5 | 30 | 快速验证 |
| `medium` | 20 | 100 | 标准评测 |
| `large` | 50 | 200 | 大规模仿真 |

### 4.4 调参示例

```powershell
# 基准 FedAvg
fl-space train --algo fedavg --dataset mnist --scale small

# FedProx + CIFAR-10 + GPU
fl-space train --algo fedprox --dataset cifar10 --scale medium --mu 0.1 --device cuda

# FedBuff 异步 + non-IID 数据
fl-space train --algo fedbuff --buffer-size 10 --non-iid --alpha 0.3

# 自定义轮次和 epoch
fl-space train --algo fedavg --rounds 200 --epochs 10 --lr 0.001

# 从 JSON 配置文件加载 + CLI 覆盖
fl-space train --config my_fl.json --device cuda --rounds 300

# 导出训练历史
fl-space train --algo fedprox -d cifar10 --output history.json
```

---

## 5. list — 查看资源

```powershell
fl-space list <资源类型>
```

| 资源类型 | 说明 | 输出内容 |
|---------|------|---------|
| `presets` | FL 实验预设 | 6 个组合预设（算法 + 规模 + 数据集） |
| `models` | 可用模型 | mlp / simplecnn |
| `satellites` | 已注册卫星类型 | polar_6, starlink_20, mixed_10, demo |
| `experiments` | 模拟实验预设 | C1~C6 六档规模配置 |

```powershell
fl-space list presets
fl-space list satellites
fl-space list experiments
```

---

## 6. export — 导出模拟结果

```powershell
fl-space export --output my_result.json [参数]
```

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--output` | `-o` | (必填) | 输出 JSON 路径 |
| `--body` | – | earth | 中心天体 (earth/mars/moon/jupiter/saturn/venus) |
| `--sats` | `-n` | 10 | 卫星数量 |
| `--stations` | `-g` | 5 | 地面站数量 |
| `--hours` | `-t` | 24 | 模拟时长 |
| `--altitude` | `-a` | 500.0 | 轨道高度 (km) |
| `--inclination` | `-i` | 90.0 | 轨道倾角 (°) |
| `--backend` | `-b` | kepler | 后端引擎 |
| `--timeslot` | – | 1.0 | 时间槽分钟数 |

**输出 JSON 包含**：配置参数、接触率、统计信息、每个卫星的完整通信记录。

```powershell
# 导出火星模拟
fl-space export --body mars --sats 5 --output mars_sim.json

# 导出大规模地球模拟
fl-space export --body earth --sats 30 --stations 13 --hours 48 --output large_earth.json
```

---

## 7. info — 系统环境

```powershell
fl-space info
```

输出示例：
```
=== SpaceFL 环境信息 ===

  框架版本: 0.1.0
  Python:   3.13.7 (CPython)
  操作系统: Windows 11
  PyTorch:  [OK] 可用
  Skyfield: [OK] 可用
  NumPy:    [OK] 2.3.5
  Matplotlib: [OK] 3.10.7
  CUDA:     [--] 不可用 (CPU only)
```

---

## 8. 外接自定义配置

### 8.1 三种配置方式

| 方式 | 适用场景 | 示例 |
|------|---------|------|
| **CLI 参数** | 快速调参、单次实验 | `fl-space simulate --sats 20 --hours 48` |
| **JSON 配置文件** | 固定配置、版本管理、团队共享 | `fl-space simulate --config my_sim.json` |
| **Python API** | 复杂逻辑、自定义组件、研究开发 | 见 `CODING_STANDARDS.md` 和项目文档 |

### 8.2 生成配置模板

```powershell
# 生成模拟器配置模板
fl-space simulate --generate-config my_sim.json

# 生成 FL 实验配置模板
fl-space train --generate-config my_fl.json
```

### 8.3 配置文件优先级

```
CLI 显式参数  >  JSON 配置文件  >  内置默认值
```

即：JSON 文件提供基础配置，CLI 参数可以覆盖 JSON 中的任意字段。

### 8.4 工作流示例

```powershell
# 1. 生成模板
fl-space simulate --generate-config my_experiment.json

# 2. 编辑 JSON 文件（修改卫星数、地面站、轨道参数等）

# 3. 运行实验，CLI 覆盖部分参数
fl-space simulate --config my_experiment.json --hours 72 --output result.json

# 4. 分析结果
python analyze.py result.json
```

---

## 9. JSON 配置文件参考

### 9.1 模拟器配置文件 (simulate/export)

```json
{
  "_comment": "SpaceFL 模拟器配置",
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
    "atmosphere_height_km": 100.0
  },
  "ground_stations": [
    {"name": "Beijing", "lat_deg": 39.9, "lon_deg": 116.4, "altitude_km": 0.05},
    {"name": "Sanya", "lat_deg": 18.25, "lon_deg": 109.5, "altitude_km": 0.0},
    {"name": "Kashi", "lat_deg": 39.5, "lon_deg": 76.0, "altitude_km": 1.3}
  ]
}
```

**字段说明：**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `num_satellites` | int | 否 | 卫星数量 (默认 5) |
| `num_ground_stations` | int | 否 | 地面站数量 (默认 3) |
| `orbit_altitude_km` | float | 否 | 轨道高度 km |
| `orbit_inclination_deg` | float | 否 | 轨道倾角 ° |
| `distribution` | str | 否 | walker / cluster / uniform |
| `backend` | str | 否 | kepler / skyfield |
| `timeslot_duration_min` | float | 否 | 每时间槽分钟数 |
| `num_timeslots` | int | 否 | 时间槽总数 |
| `body` | dict | 否 | 天体参数 (name/radius_km/GM/rotation_period_hours) |
| `ground_stations` | list | 否 | 地面站列表 (name/lat_deg/lon_deg/altitude_km) |

> 地面站也支持简写格式：`["Beijing", 39.9, 116.4]`

### 9.2 FL 实验配置文件 (train)

```json
{
  "_comment": "SpaceFL FL 实验配置",
  "algorithm": "fedavg",
  "num_rounds": 50,
  "num_clients": 10,
  "fraction": 0.5,
  "local_epochs": 5,
  "batch_size": 32,
  "learning_rate": 0.01,
  "mu": 0.01,
  "buffer_size": 5,
  "staleness_weight": false,
  "device": "cpu",
  "seed": 42
}
```

**字段说明：**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `algorithm` | str | ✅ | fedavg / fedprox / fedbuff |
| `num_rounds` | int | 否 | 全局训练轮次 (默认 50) |
| `num_clients` | int | 否 | 客户端总数 (默认 10) |
| `fraction` | float | 否 | 每轮参与客户端比例 (默认 0.5) |
| `local_epochs` | int | 否 | 本地训练 epoch (默认 5) |
| `batch_size` | int | 否 | 训练 batch size (默认 32) |
| `learning_rate` | float | 否 | 学习率 (默认 0.01) |
| `mu` | float | 否 | FedProx 近端项系数 (默认 0.01) |
| `buffer_size` | int | 否 | FedBuff 缓冲区大小 (默认 5) |
| `staleness_weight` | bool | 否 | FedBuff 陈旧度降权 (默认 false) |
| `device` | str | 否 | cpu / cuda (默认 cpu) |
| `seed` | int | 否 | 随机种子 |

### 9.3 通过 Python API 接入自定义逻辑

对于超出 JSON 配置能力的场景（如自定义 FL 组件、自定义模型），使用 Python API：

```python
# ── 接入自定义 FL 组件 ──
from fl_space.fl.core import LocalTrainer, ClientUpdate
from fl_space.fl.server import FLServer, FLConfig
from fl_space.fl.runner import FLRunner

class MyCustomTrainer(LocalTrainer):
    """自定义本地训练逻辑。"""
    def train(self, client_id, model, train_loader, global_weights, round_num, **kwargs):
        # 你的训练逻辑
        ...
        return ClientUpdate(...)

# 使用自定义组件
config = FLConfig(algorithm="fedavg", num_rounds=100)
runner = FLRunner(config, MyRandomSelector(), MyCustomTrainer(), ...)
history = runner.run()

# ── 接入自定义模型 ──
from fl_space.fl.models import register_model, get_model
import torch.nn as nn

class MyResNet(nn.Module):
    ...

register_model("my_resnet", MyResNet)
model = get_model("my_resnet", num_classes=100)

# ── 自定义地面站网络 ──
from fl_space.environment import GroundStation, GroundStationNetwork

stations = GroundStationNetwork([
    GroundStation(name="MySite", lat_deg=30.0, lon_deg=120.0),
    GroundStation(name="MySite2", lat_deg=-30.0, lon_deg=-60.0),
])

from fl_space.simulator import OrbitSimulator
sim = OrbitSimulator(
    ground_station_network=stations,
    num_satellites=10,
    ...
)
```

---

## 附录：常见问题

**Q: 如何分享实验配置给他人？**
```powershell
# 生成配置文件
fl-space train --generate-config shared_experiment.json
# 编辑后提交到 git，他人可直接使用
fl-space train --config shared_experiment.json
```

**Q: 配置文件中某些字段不填会怎样？**
使用内置默认值。只填需要覆盖的字段即可。

**Q: 如何同时看到 JSON 配置和 CLI 覆盖的效果？**
运行时会在输出中标注 `[配置] 加载 xxx.json`，打印的参数是最终生效的值。

**Q: 如何批量运行多组实验？**
Windows 下可使用批处理脚本：
```batch
fl-space simulate --config base.json --sats 10 --output exp1.json
fl-space simulate --config base.json --sats 20 --output exp2.json
fl-space simulate --config base.json --sats 30 --output exp3.json
```
