# SpaceFL 项目记忆

## 项目概述
- **项目**: 太空联邦学习 (SpaceFL) 开源研究框架
- **基础工作区**: `D:\project_wang\fl_space_reproduction1` (原论文复现项目)
- **新框架工作区**: `d:\fl_space` (模块化重构)
- **目标**: 将太空FL做成可配置、可扩展的开源研究框架

## 开发计划 (7步)
1. ✅ 环境模拟设置 — 已完成模块化重构 (Kepler + Skyfield双后端, ruff规范化)
2. ✅ 卫星设置（多星簇、自定义注册、可视化）— 2026-05-29
3. 太空联邦的算法选择
4. 数据集
5. 地面基站设置
6. 保存节点（checkpoint机制）
7. 信息展示增强

## Step 1 完成内容 (d:\fl_space)

### 四层模块化架构
- **environment/**: CelestialBody（地球/火星/月球/木星/土星/金星/自定义 + JPL精确参数）、AtmosphereModel（无/恒定/指数）、GroundStation/GroundStationNetwork、coordinate_utils
- **orbit/**: KeplerOrbit（圆形/椭圆/任意倾角）、ConstellationConfig（Walker/星簇/均匀分布）、VisibilityEngine（单星/多星批量）、**SkyfieldOrbitBackend**（SGP4高精度后端）
- **simulator/**: OrbitSimulator（主模拟器，双后端kepler/skyfield）、ContactMatrix（兼容模式+完整模式）
- **config/**: 预设配置（天体/星座/地面站/实验）、配置加载器

### 双后端支持
- `backend="kepler"` (默认): 轻量开普勒二体力学，无外部依赖，计算快
- `backend="skyfield"`: 高精度SGP4传播 + JPL DE421星历，需 `pip install skyfield`
  - 使用 sgp4.Satrec 底层API（避免合成TLE格式问题）
  - 实现eci-to-geodetic坐标转换
  - Topocentric仰角/方位角计算
  - 支持TLE/SGP4 + Kepler要素双输入

### 第三方库调研结论
- **Skyfield** (MIT, 纯Python): ✅ 最佳选择 — 地面站可见性 + JPL星历 + SGP4
- **AstroLib** (C++): ❌ 不可直接集成 — C++静态库，仅Windows+VS，但算法可参考
- **hapsira/poliastro** (MIT, Python): ⚠️ 缺地面站可见性模块
- **Orekit** (Java+Py): ❌ 依赖过重

### 关键特性
- 可自定义行星（默认地球），内置6个天体预设 + JPL精确参数
- 完整接触模式记录所有可见地面站（满足需求：每个卫星与哪些基站传输了信息）
- 兼容原 orbit_sim_v2.py 接口
- 50+ 项自动化测试全部通过（含Skyfield后端）

### 代码规范化 (2026-05-29)
- **工具**: Ruff 0.15.15 (linting + formatting)，配置在 `pyproject.toml` `[tool.ruff]`
- **命令**: `ruff check fl_space/` (检查), `ruff check --fix` (修复), `ruff format fl_space/` (格式化)
- **忽略规则**: RUF002 (中文全角标点), RUF003 (中文注释全角), N806 (轨道力学符号M/E/N/GM)
- **命名例外**: GM, M, M0, E, dE, cos_E, sin_E, N (标准轨道力学符号)
- **行宽**: 100 (科研代码适度放宽)
- 所有 `__init__.py` 必须包含 `__all__` 列表和显式 `as` 重导出

### 技术约定
- Python 3.9+, numpy + 可选 skyfield
- 用 `verbose=False` 控制大规模星座的输出
- sgp4init的no_kozai参数单位为 rad/min（不是rev/day）
- 类型注解用 Python 3.9+ 风格: `list[X]` (非 `List[X]`), `X | None` (非 `Optional[X]`)

## Step 2 完成内容 (2026-05-29)

### 卫星配置系统 (`fl_space/orbit/satellite_config.py`)
- **MultiClusterConfig**: 多星簇星座配置，支持多个不同高度/倾角的星簇
- **ClusterSpec**: 单星簇规格（卫星数、高度、倾角、分布策略、RAAN偏移）
- **SatelliteSpec**: 单星精细配置（完整轨道六要素 + 通信参数）
- 内置预设：`polar_only()`, `starlink_like()`, `mixed_orbit()`, `demo_default()`
- 支持 `to_dict()`/`from_dict()` 序列化

### 卫星注册表 (`fl_space/orbit/satellite_registry.py`)
- `SatelliteRegistry` 全局单例 + `registry` 实例
- 三种注册方式：装饰器 `@registry.register()`、函数 `registry.register_func()`、字典 `registry.register_from_dict()`
- 内置4种卫星类型：polar_6, starlink_20, mixed_10, demo
- 支持 `list_types()` 查看所有已注册类型

### 轨道可视化 (`fl_space/viz/`)
- `OrbitVisualizer`: 2D地图投影（简化大陆轮廓）+ 星下点轨迹 + 地面站
- `plot_dashboard()`: 综合仪表盘（地图 + 统计 + 接触矩阵热力图）
- `quick_plot()`: 一键传入模拟器生成完整可视化
- 深色主题（#0d1117 背景），6种对比色系
- 纯 matplotlib 实现，无 cartopy 依赖

### OrbitSimulator 增强
- 新增 `orbits` 参数：支持直接传入预构建轨道列表（优先级最高）

### Demo 输出 (examples/output/)
- demo1: 极轨4+LEO壳6 双星簇 → 接触率21.7%
- demo2: 自定义注册卫星类型 → 接触率18.5%
- demo3: Walker壳+自定义单星(SSO+赤道) → 接触率27.3%
- demo4: 4种星座对比图
- demo5: 单星星下点轨迹+接触高亮

## 代码规范文档 (2026-06-01)
- 创建了 `CODING_STANDARDS.md` — 完整的代码规范与开发指南
- 包含11个章节：架构概述、命名、类型注解、文档字符串、导入、数据模式、错误处理、Ruff配置、测试、AI辅助开发准则、检查清单
- 特别包含 AI 辅助开发准则（第10章），涵盖提示词模板、AI代码检查流程、严禁事项
- 附录提供常见模式速查和常见错误示例

