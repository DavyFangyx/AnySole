# E6 补充表：router 的 T 分支权重 k（部位 × 遮挡）

> 补充材料（历史）：本表为 full competitive router 的 k 分析（新名 `V8_IMU_Router_Full`），不进正文主表；正文竞争式路由用 router-only（`V8_IMU_Router`）。主模型为 `V8_IMU_Residual`。

口径：`router_ft_enum_fS5S6S11.pt`；test `fS5S6S11`；窗口 stride 20；四档使用相同运动窗，只替换视觉缓存。
k = σ(s_v − s_t)，k 越大 T 分支权重越大；V 分支权重为 1−k。窗间标准差按窗均值计算，窗内标准差为各窗时间标准差的平均。

| 组 | clean | occ020 | occ050 | occ100 |
|---|---:|---:|---:|---:|
| root velocity | 0.079 | 0.096 | 0.175 | 0.979 |
| root pose + torso/head/arms | 0.418 | 0.468 | 0.629 | 0.971 |
| left leg + foot | 0.656 | 0.710 | 0.852 | 0.979 |
| right leg + foot | 0.654 | 0.708 | 0.864 | 0.982 |

n_windows = 1221；n_frames = 24420（每档）。
