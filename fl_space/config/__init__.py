"""
配置层 — 默认参数预设和配置加载

主要导出:
    defaults    — 天体/星座/地面站/实验预设
    loader      — 从 JSON/YAML/字典加载配置
"""

from . import defaults as defaults
from .loader import (
    load_celestial_body as load_celestial_body,
)
from .loader import (
    load_constellation_config as load_constellation_config,
)
from .loader import (
    load_ground_stations as load_ground_stations,
)
from .loader import (
    load_sim_config_from_dict as load_sim_config_from_dict,
)
from .loader import (
    load_sim_config_from_json as load_sim_config_from_json,
)

__all__ = [
    "defaults",
    "load_celestial_body",
    "load_constellation_config",
    "load_ground_stations",
    "load_sim_config_from_dict",
    "load_sim_config_from_json",
]
