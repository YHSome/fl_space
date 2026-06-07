"""
外部框架集成模块 — 可插拔的外部生态适配器。

提供：
    - flower/ — Flower 联邦学习框架适配器
    - (future) ray/ — Ray 分布式训练适配器
    - (future) tensorflow/ — TensorFlow Federated 适配器

设计原则：
    - 每个集成是独立的可选模块
    - 不修改 SpaceFL 核心代码
    - 通过适配器模式转换数据结构
"""
