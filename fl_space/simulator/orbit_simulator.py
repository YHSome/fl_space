"""
轨道模拟器 — 模块化的太空联邦学习环境模拟

整合了环境层、轨道层、接触矩阵层的所有功能，
提供统一的、参数化的模拟接口。

支持:
    - 自定义行星（地球/火星/月球/自定义）
    - 灵活的星座配置（Walker/星簇/均匀）
    - 两种接触模式（兼容/完整）
    - 可扩展的大气模型
    - 地面站最小仰角约束

与原 orbit_sim_v2.py 兼容，同时提供更丰富的功能。
"""

import time as _time
from typing import Optional

import numpy as np

from fl_space.environment import (
    CelestialBody,
    GroundStationNetwork,
    create_default_network,
    create_extended_network,
)
from fl_space.orbit import (
    ConstellationConfig,
    KeplerOrbit,
    MultiSatVisibility,
    generate_orbits,
)

from .contact_matrix import ContactMatrix


class OrbitSimulator:
    """
    模块化轨道接触模拟器。

    双后端支持:
        - backend="kepler"  : 轻量开普勒力学（默认，无外部依赖）
        - backend="skyfield": 高精度 SGP4/TLE + JPL 星历（需 pip install skyfield）

    使用示例::

        from fl_space.simulator import OrbitSimulator

        # 默认配置: Kepler后端 + 地球 + 3卫星 + 2地面站
        sim = OrbitSimulator()
        print(sim.summary())

        # Mars + Walker constellation
        from fl_space.environment import CelestialBody
        mars = CelestialBody.mars()
        sim_mars = OrbitSimulator(body=mars, num_satellites=5, num_ground_stations=3)

        # Skyfield高精度后端（地面站可见性 + TLE支持）
        sim_sf = OrbitSimulator(
            backend="skyfield", num_satellites=5, num_ground_stations=3,
            sim_start_date=(2024, 6, 1),
        )

        # 查询接触
        contacts = sim.get_satellites_in_contact(60)
        detail = sim.get_contact_detail(sat_id=0, timeslot=60)

    Parameters
    ----------
    backend : str
        "kepler" (默认) | "skyfield"。Skyfield 提供更精确的 SGP4 传播和
        地面站可见性计算，但需要 pip install skyfield。
    sim_start_date : tuple, optional
        (year, month, day) — 模拟开始日期，仅 skyfield 后端需要。
        默认 (2024, 6, 1)。
    """

    def __init__(
        self,
        # 天体参数
        body: Optional[CelestialBody] = None,

        # 星座参数
        num_satellites: int = 3,
        num_ground_stations: int = 2,
        orbit_altitude_km: float = 500.0,
        orbit_inclination_deg: float = 90.0,

        # 预构建轨道（优先级最高）
        orbits: Optional[list[KeplerOrbit]] = None,

        # 星座分布
        constellation_config: Optional[ConstellationConfig] = None,
        distribution: str = "uniform",  # "walker" | "cluster" | "uniform"

        # 地面站
        ground_station_network: Optional[GroundStationNetwork] = None,
        use_extended_gs: bool = False,

        # 时间参数
        timeslot_duration_min: float = 1.0,
        num_timeslots: int = 1440,

        # 轨道后端选择
        backend: str = "kepler",  # "kepler" | "skyfield"
        sim_start_date: Optional[tuple[int, int, int]] = None,  # (year, month, day) for skyfield

        # 接触矩阵模式
        contact_mode: str = "full",  # "simple" | "full"

        # 其他
        random_seed: int = 42,
        verbose: bool = True,
    ):
        """
        Parameters
        ----------
        body : CelestialBody, optional
            中心天体。默认地球。
        num_satellites : int
            卫星数量。
        num_ground_stations : int
            地面站数量。
        orbit_altitude_km : float
            轨道高度 (km)。
        orbit_inclination_deg : float
            轨道倾角 (°)。
        constellation_config : ConstellationConfig, optional
            星座配置（优先于 num_satellites/orbit_altitude_km）。
        distribution : str
            分布策略: "walker" / "cluster" / "uniform"。
        ground_station_network : GroundStationNetwork, optional
            自定义地面站网络（优先于 num_ground_stations）。
        use_extended_gs : bool
            是否使用扩展地面站（准全球覆盖）。
        timeslot_duration_min : float
            每个 timeslot 时长 (分钟)。
        num_timeslots : int
            timeslot 总数。
        contact_mode : str
            "simple" 兼容模式 / "full" 完整记录。
        random_seed : int
            随机种子。
        verbose : bool
            是否打印进度信息。
        """
        self.rng = np.random.RandomState(random_seed)
        self.verbose = verbose
        self.backend_mode = backend
        self.sim_start_date = sim_start_date or (2024, 6, 1)

        # ---- 天体 ----
        self.body = body if body is not None else CelestialBody.earth()

        # ---- 星座配置 ----(仅当未直接提供 orbits 时使用)----
        if orbits is not None:
            # 使用预构建轨道
            self.orbits = list(orbits)
            self.num_satellites = len(self.orbits)
            self.constellation_config = None
            # 从轨道推断平均参数
            if self.orbits:
                avg_alt = sum(o.elements.semi_major_axis_km - self.body.radius_km
                              for o in self.orbits) / len(self.orbits)
                self.orbit_altitude_km = avg_alt
                self.orbit_inclination_deg = self.orbits[0].elements.inclination_deg
            else:
                self.orbit_altitude_km = orbit_altitude_km
                self.orbit_inclination_deg = orbit_inclination_deg
        elif constellation_config is not None:
            self.constellation_config = constellation_config
            self.num_satellites = constellation_config.num_satellites
            self.orbit_altitude_km = constellation_config.altitude_km
            self.orbit_inclination_deg = constellation_config.inclination_deg
            # ---- 生成轨道 ----
            self.orbits: list[KeplerOrbit] = generate_orbits(
                self.constellation_config, self.body
            )
        else:
            self.constellation_config = ConstellationConfig(
                num_satellites=num_satellites,
                num_planes=1,
                inclination_deg=orbit_inclination_deg,
                altitude_km=orbit_altitude_km,
                distribution=distribution,
            )
            self.num_satellites = self.constellation_config.num_satellites
            self.orbit_altitude_km = self.constellation_config.altitude_km
            self.orbit_inclination_deg = self.constellation_config.inclination_deg
            # ---- 生成轨道 ----
            self.orbits: list[KeplerOrbit] = generate_orbits(
                self.constellation_config, self.body
            )

        # ---- 地面站 ----
        if ground_station_network is not None:
            self.ground_network = ground_station_network
        elif use_extended_gs:
            self.ground_network = create_extended_network(num_ground_stations)
        else:
            self.ground_network = create_default_network(num_ground_stations)

        self.num_ground_stations = self.ground_network.count

        # ---- 时间参数 ----
        self.timeslot_duration_min = timeslot_duration_min
        self.num_timeslots = num_timeslots

        # 轨道周期 (取第一颗卫星，因为都使用相同高度)
        self.orbit_period_min = self.orbits[0].period_min if self.orbits else 0.0

        # ---- 轨道后端 ----
        self._sf_backend = None
        self._sf_satellites = []
        if backend == "skyfield":
            self._init_skyfield_backend()

        # ---- 接触矩阵 ----
        self.contact_mode = contact_mode
        self.contact_matrix = ContactMatrix(
            self.num_satellites, self.num_timeslots, mode=contact_mode
        )

        # ---- 生成接触数据 ----
        t0 = _time.time()
        self._generate_contacts()
        self._gen_time = _time.time() - t0

        # ---- 统计 ----
        self.stats = self.contact_matrix.compute_statistics()

    # ============================================================
    # Skyfield 后端初始化
    # ============================================================

    def _init_skyfield_backend(self):
        """初始化 Skyfield 高精度后端。"""
        from fl_space.orbit.skyfield_backend import SkyfieldOrbitBackend

        self._sf_backend = SkyfieldOrbitBackend()
        self._sf_satellites = []

        # 计算模拟起始时刻的儒略日（TLE epoch 需与模拟时间一致）
        from datetime import datetime
        yr, mo, dy = self.sim_start_date
        datetime(yr, mo, dy)
        # 转为 julian date (简化: datetime to JD)
        a = (14 - mo) // 12
        y = yr + 4800 - a
        m = mo + 12 * a - 3
        epoch_jd = (dy + (153 * m + 2) // 5 + 365 * y + y // 4
                     - y // 100 + y // 400 - 32045 - 0.5)

        for i, orbit in enumerate(self.orbits):
            sat = self._sf_backend.create_satellite_from_kepler(
                altitude_km=self.orbit_altitude_km,
                inclination_deg=orbit.elements.inclination_deg,
                raan_deg=orbit.elements.raan_deg,
                true_anomaly_deg=orbit.elements.true_anomaly_deg,
                eccentricity=orbit.elements.eccentricity,
                epoch_jd=epoch_jd,
                name=f"SF-SAT-{i}",
            )
            self._sf_satellites.append(sat)

        if self.verbose:
            print(f"    [OrbitSim] Skyfield后端初始化完成 "
                  f"({self.num_satellites} satellites, "
                  f"epoch {self.sim_start_date}, JPL DE421 ephemeris)")

    # ============================================================
    # 接触矩阵生成
    # ============================================================

    def _generate_contacts(self):
        """生成完整接触矩阵（根据后端自动选择算法）。"""
        if self.backend_mode == "skyfield" and self._sf_backend is not None:
            self._generate_contacts_skyfield()
        else:
            self._generate_contacts_kepler()

    def _generate_contacts_kepler(self):
        """Kepler 后端接触矩阵生成。"""
        multi_vis = MultiSatVisibility(self.body, self.orbits, self.ground_network)

        for ts in range(self.num_timeslots):
            if self.verbose and self.num_timeslots > 5000 and ts % (self.num_timeslots // 10) == 0:
                print(f"    [OrbitSim] 生成接触数据: {ts}/{self.num_timeslots} "
                      f"({100*ts/self.num_timeslots:.0f}%)", flush=True)

            time_min = ts * self.timeslot_duration_min
            all_visible = multi_vis.visible_matrix_at_time(time_min)

            for sat_id, visible_gs in enumerate(all_visible):
                self.contact_matrix.set_contacts(sat_id, ts, visible_gs)

        if self.verbose:
            print(f"    [OrbitSim:kepler] 接触矩阵生成完毕, "
                  f"接触率: {np.mean(np.sum(self.contact_matrix.simple_matrix >= 0, axis=1)) / self.num_timeslots * 100:.1f}%")

    def _generate_contacts_skyfield(self):
        """Skyfield 后端接触矩阵生成（使用精确仰角判断）。"""
        sf = self._sf_backend
        yr, mo, dy = self.sim_start_date

        for ts in range(self.num_timeslots):
            if self.verbose and self.num_timeslots > 1000 and ts % (self.num_timeslots // 10) == 0:
                print(f"    [OrbitSim:skyfield] 生成接触数据: {ts}/{self.num_timeslots} "
                      f"({100*ts/self.num_timeslots:.0f}%)", flush=True)

            minute_of_day = ts * self.timeslot_duration_min
            day_offset = int(minute_of_day / 1440)
            minute_in_day = minute_of_day % 1440
            hour = minute_in_day / 60.0

            # 计算当前UTC日期
            from datetime import datetime, timedelta
            current_dt = datetime(yr, mo, dy) + timedelta(days=day_offset)
            cy, cm, cd = current_dt.year, current_dt.month, current_dt.day

            for sat_id, sat in enumerate(self._sf_satellites):
                visible_gs = []
                for gs_id, gs in enumerate(self.ground_network):
                    try:
                        elev, _, _ = sf.elevation_at(
                            sat, gs.lat_deg, gs.lon_deg,
                            cy, cm, cd, hour,
                        )
                        if elev >= gs.min_elevation_deg:
                            visible_gs.append(gs_id)
                    except Exception:
                        # Skip if SGP4 propagation fails for this time
                        pass

                self.contact_matrix.set_contacts(sat_id, ts, visible_gs)

        if self.verbose:
            print(f"    [OrbitSim:skyfield] 接触矩阵生成完毕, "
                  f"接触率: {np.mean(np.sum(self.contact_matrix.simple_matrix >= 0, axis=1)) / self.num_timeslots * 100:.1f}%")

    # ============================================================
    # 查询接口（兼容原 orbit_sim_v2.py）
    # ============================================================

    def get_contact_at_timeslot(self, sat_id: int, timeslot: int) -> int:
        """获取简单接触 (gs_id 或 -1)，兼容原接口。"""
        return self.contact_matrix.get_first_contact(sat_id, timeslot)

    def get_all_contacts(self, sat_id: int, timeslot: int) -> list[int]:
        """获取所有可见地面站 ID 列表。"""
        return self.contact_matrix.get_all_contacts(sat_id, timeslot)

    def get_next_contact(
        self, sat_id: int, after_timeslot: int
    ) -> Optional[tuple[int, int]]:
        """获取下一次接触窗口。"""
        return self.contact_matrix.get_next_contact(sat_id, after_timeslot)

    def get_satellites_in_contact(self, timeslot: int) -> list[int]:
        """获取某时刻可通信的所有卫星。"""
        return self.contact_matrix.get_satellites_in_contact(timeslot)

    def get_contact_detail(self, sat_id: int, timeslot: int) -> dict:
        """获取某时刻某卫星的接触详情。"""
        return self.contact_matrix.get_contact_detail(
            sat_id, timeslot, gs_names=self.ground_network.names
        )

    # ============================================================
    # FL 轮次计算
    # ============================================================

    def compute_round_duration_sync(
        self,
        local_epochs: int = 1,
        compute_time_per_epoch_min: float = 2.0,
        start_timeslot: int = 0,
    ) -> tuple[int, int]:
        """
        计算同步 FL 一轮的持续时间。

        同步机制: 所有卫星必须都完成训练并返回模型后才算一轮结束。

        Parameters
        ----------
        local_epochs : int
            本地训练 epoch 数。
        compute_time_per_epoch_min : float
            每 epoch 计算时间 (分钟)。
        start_timeslot : int
            起始 timeslot。

        Returns
        -------
        (round_duration_slots, last_return_timeslot)
        """
        train_time_slots = int(
            local_epochs * compute_time_per_epoch_min / self.timeslot_duration_min
        )

        return_times = []
        for sat_id in range(self.num_satellites):
            # 1) 第一次接触 (接收全局模型)
            first = self.get_next_contact(sat_id, start_timeslot - 1)
            if first is None:
                continue
            recv_ts, _ = first

            # 2) 训练结束后找返回接触
            train_end_ts = recv_ts + train_time_slots
            ret = self.get_next_contact(sat_id, train_end_ts - 1)
            if ret is None:
                continue
            return_ts, _ = ret
            return_times.append(return_ts)

        if not return_times:
            return (0, start_timeslot)

        last_ts = max(return_times)
        duration = last_ts - start_timeslot
        return (duration, last_ts)

    # ============================================================
    # 额外查询（新的增强接口）
    # ============================================================

    def get_sat_position(self, sat_id: int, timeslot: int) -> tuple[float, float]:
        """
        获取卫星在指定 timeslot 的星下点位置。

        Returns
        -------
        (lat_deg, lon_deg)
        """
        time_min = timeslot * self.timeslot_duration_min
        return self.orbits[sat_id].position_at_time_deg(time_min)

    def get_sat_positions_at_timeslot(self, timeslot: int) -> dict[int, tuple[float, float]]:
        """获取某时刻所有卫星位置。"""
        return {sat_id: self.get_sat_position(sat_id, timeslot)
                for sat_id in range(self.num_satellites)}

    def get_sat_trajectory(
        self, sat_id: int, timeslots: Optional[list[int]] = None
    ) -> list[tuple[float, float]]:
        """
        获取卫星轨迹（一系列星下点）。

        Parameters
        ----------
        sat_id : int
            卫星 ID。
        timeslots : List[int], optional
            时刻列表，默认所有 timeslot。

        Returns
        -------
        List[Tuple[float, float]]
            [(lat_deg, lon_deg), ...]
        """
        if timeslots is None:
            timeslots = list(range(self.num_timeslots))
        return [self.get_sat_position(sat_id, ts) for ts in timeslots]

    def get_communication_record(
        self, sat_id: int
    ) -> list[dict]:
        """
        获取某卫星的完整通信记录。

        满足用户需求: "每个卫星与哪些基站传输了信息，这些也要记录下来"

        Returns
        -------
        List[Dict]
            每项含 {timeslot, gs_ids, gs_names, sat_position}。
        """
        records = []
        for ts in range(self.num_timeslots):
            gs_ids = self.get_all_contacts(sat_id, ts)
            if gs_ids:
                records.append({
                    'timeslot': ts,
                    'time_min': ts * self.timeslot_duration_min,
                    'gs_ids': gs_ids,
                    'gs_names': [self.ground_network.names[gid] for gid in gs_ids
                                 if 0 <= gid < len(self.ground_network.names)],
                    'sat_position_deg': self.get_sat_position(sat_id, ts),
                })
        return records

    # ============================================================
    # 摘要与导出
    # ============================================================

    def summary(self) -> str:
        """返回模拟器配置摘要。"""
        backend_label = {
            'kepler': 'Kepler (轻量开普勒力学)',
            'skyfield': 'Skyfield (SGP4 + JPL DE421精确星历)',
        }.get(self.backend_mode, self.backend_mode)

        lines = [
            "=" * 60,
            "  SpaceFL 轨道模拟器 配置摘要",
            "=" * 60,
            f"  轨道后端:           {backend_label}",
            f"  中心天体:           {self.body.name}",
            f"    半径:              {self.body.radius_km} km",
            f"    GM:                {self.body.GM:.1f} km^3/s^2",
            f"    自转周期:          {self.body.rotation_period_hours} h",
            f"    大气层高度:        {self.body.atmosphere_height_km} km",
            "",
            f"  卫星数量:           {self.num_satellites}",
            f"  轨道高度:           {self.orbit_altitude_km} km",
            f"  轨道倾角:           {self.orbit_inclination_deg}°",
            f"  轨道周期:           {self.orbit_period_min:.1f} min",
            f"  分布策略:           {self.constellation_config.distribution if self.constellation_config else '自定义轨道'}",
            "",
            f"  地面站数量:         {self.num_ground_stations}",
            f"  Timeslot粒度:       {self.timeslot_duration_min} min",
            f"  总Timeslot数:       {self.num_timeslots}",
            f"  模拟时长:           {self.num_timeslots * self.timeslot_duration_min / 60:.1f} h",
            f"  接触矩阵模式:       {self.contact_mode}",
            f"  模拟起始日期:       {self.sim_start_date}",
            f"  星历来源:           {'JPL DE421 (Skyfield)' if self.backend_mode == 'skyfield' else '开普勒二体'}",
            "",
            "  地面站列表:",
        ]

        for i, gs in enumerate(self.ground_network):
            lines.append(
                f"    [{i}] {gs.name}: "
                f"({gs.lat_deg:+.2f}°, {gs.lon_deg:+.2f}°)"
                + (f" min_el={gs.min_elevation_deg}°" if gs.min_elevation_deg > 0 else "")
            )

        lines.extend([
            "",
            "  接触统计:",
            f"    总接触次数:        {self.stats['total_contacts']}",
            f"    每卫星平均接触:    {self.stats['avg_contacts_per_sat']:.1f}",
            f"    总体接触率:        {self.stats['contact_rate']*100:.1f}%",
            f"    生成耗时:          {self._gen_time:.1f}s",
            "=" * 60,
        ])

        return "\n".join(lines)

    def export(self, filepath: str):
        """导出接触矩阵等完整数据到 JSON。"""
        self.contact_matrix.save_json(filepath, gs_names=self.ground_network.names)

    def export_contact_matrix(self, filepath: str):
        """兼容原 orbit_sim_v2.py 的导出接口。"""
        import json
        data = {
            'config': {
                'body': self.body.to_dict(),
                'num_satellites': self.num_satellites,
                'num_ground_stations': self.num_ground_stations,
                'orbit_altitude_km': self.orbit_altitude_km,
                'orbit_period_min': self.orbit_period_min,
                'timeslot_duration_min': self.timeslot_duration_min,
                'num_timeslots': self.num_timeslots,
                'distribution': self.constellation_config.distribution,
            },
            'ground_stations': [
                {'id': i, 'name': gs.name, 'lat': gs.lat_deg, 'lon': gs.lon_deg}
                for i, gs in enumerate(self.ground_network)
            ],
            'contact_matrix_simple': self.contact_matrix.simple_matrix.tolist(),
            'statistics': self.stats,
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def __repr__(self) -> str:
        return (
            f"OrbitSimulator(backend={self.backend_mode}, body={self.body.name}, "
            f"{self.num_satellites}sats x {self.num_ground_stations}gs, "
            f"alt={self.orbit_altitude_km}km, "
            f"contact_rate={self.stats['contact_rate']:.1%})"
        )


# ============================================================
# 工厂函数
# ============================================================

def create_default_simulator() -> OrbitSimulator:
    """
    创建默认配置模拟器（与原 orbit_sim_v2.py 兼容）。

    Returns
    -------
    OrbitSimulator
        地球 + 3卫星 + 2地面站，24小时模拟。
    """
    return OrbitSimulator(
        body=CelestialBody.earth(),
        num_satellites=3,
        num_ground_stations=2,
        orbit_altitude_km=500.0,
        orbit_inclination_deg=90.0,
        distribution="uniform",
        timeslot_duration_min=1.0,
        num_timeslots=1440,
        contact_mode="full",
        random_seed=42,
    )


def create_mars_simulator() -> OrbitSimulator:
    """创建火星模拟器示例。"""
    from fl_space.environment.ground_station import GroundStation, GroundStationNetwork

    # 火星地面站（假设的未来基站位置）
    mars_gs = GroundStationNetwork([
        GroundStation("Olympus Base", 18.65, -133.8),       # Olympus Mons
        GroundStation("Valles Marineris", -14.0, -59.0),    # 水手谷
        GroundStation("Gale Crater", -4.6, 137.4),          # 好奇号着陆点
        GroundStation("Jezero Crater", 18.4, 77.6),         # 毅力号着陆点
        GroundStation("Utopia Planitia", 49.7, 118.0),      # 祝融号着陆点
    ])

    return OrbitSimulator(
        body=CelestialBody.mars(),
        constellation_config=ConstellationConfig(
            num_satellites=5,
            num_planes=2,
            inclination_deg=60.0,
            altitude_km=300.0,
            distribution="walker",
            phasing_factor=1,
        ),
        ground_station_network=mars_gs,
        timeslot_duration_min=1.0,
        num_timeslots=1440,
        contact_mode="full",
    )
