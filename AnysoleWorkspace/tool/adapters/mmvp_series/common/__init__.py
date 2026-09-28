"""MMVP 数据端公共件：canonical 路径、manifest/split 读取、逐 session 事实。

只读消费 ``shared/`` 与 ``protocol/``；本模块不写任何 tree。各模型的
adapter 根目录（``model-input://<model>/...``）由各自模块自行定义。
"""
