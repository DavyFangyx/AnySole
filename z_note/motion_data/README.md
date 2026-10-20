# motion_data：BVH × SMPL 趾间隙异常排查包（2026-10-15）

来源：D_Test3 修复验证中发现的异常 session 的**原文件** + SMPL 模板。
每个 `<sid>/` 目录 = 该 session 的 BVH（`*_SkeletonN.bvh`，原始 cm/y-up）与
SMPL（`motion_neutral_smpl.npz`，neutral 身体模型拟合，m/y-up）；`smpl_model/` =
`SMPL_NEUTRAL.pkl`（项目唯一 SMPL 模板，smplx 正演与 FK 共用）。

## 口径（读数值前必看）

- **帧号**：全部指 facts 40Hz 视觉帧轴（0 起），对应 session 的 `frames.npz`；
- **高度 h**：各自源的统计地面之上。BVH 地面 = 该 BVH 全部关节 z（z-up 显示
  转换后）的 5 分位；SMPL 地面 = `estimate_floor_y`（全部 SMPL 关节 y 的 5 分位，
  共享实现）。负 h = 足底/趾尖压入统计线以下，正常；
- **BVH 足三点**：`*Foot`=踝（离地 ~10cm）、`*ToeBase`=趾根、`*ToeBase_EndSite*`
  =趾尖（无旋转通道，比趾根低 3–4cm，`foot_signal` 判的就是它）；
- **SMPL 足**：关节只有 ankle + foot（foot 在趾根处），趾尖由网格顶点承载
  （`smpl_contact` 判据 = 足底顶点离地 <2cm）；
- **bvh_h 判据**：BVH 足最低点（Foot/ToeBase/EndSite 的 min z）<5cm 判接触，
  4/6cm 迟滞。`smpl_contact`：足底顶点 <2cm 判接触。两者都判"足未真正离地"。

## 异常清单（bvh_h 接近全接触的 9 个 session）

| Session | 异常类型 | BVH 足最低点 max | BVH 踝 max | SMPL 踝 max | bvh_h L/R |
|---|---|---|---|---|---|
| S11053（S11 动作05，Skeleton2） | **高速摆腿 + 极低趾间隙** | L 4.0cm@f237 / R 5.0cm@f224 | 0.231m@f237 / 0.243m@f65 | 0.175m@f236 / 0.185m@f64 | 1.00/1.00 |
| S5091（S5 动作09，Skeleton3） | 趾下垂摆动典型 | L 6.2cm@f366 / R 3.7cm@f421 | 0.188m@f316 / 0.189m@f224 | 0.146m@f316 / 0.128m@f224 | 0.99/1.00 |
| S11032（S11 动作03，Skeleton2） | 踝摆 25cm 而趾尖仅 4.8cm | L 4.8cm@f327 / R 5.4cm@f60 | 0.252m@f84 / 0.254m@f106 | 0.204m@f83 / 0.204m@f60 | 1.00/1.00 |
| S11073（S11 动作07，Skeleton2） | **对照组**：唯一越过 6cm | L 6.5cm@f65 / R 6.3cm@f44 | 0.231m@f64 / 0.232m@f44 | 0.179m@f64 / 0.168m@f64 | 0.97/0.95 |
| S11011（S11 动作01，Skeleton2） | 足最低点 7–8mm | L 0.7cm@f483 / R 0.8cm@f34 | 0.134m@f263 / 0.120m@f43 | 0.076m@f263 / 0.064m@f42 | 1.00/1.00 |
| S10101（S10 动作10，Skeleton1） | 足最低点 3–5mm（拖步） | L 0.3cm@f65 / R 0.5cm@f65 | 0.111m@f297 / 0.114m@f121 | 0.093m@f419 / 0.089m@f175 | 1.00/1.00 |
| S10103（S10 动作10，Skeleton1） | 同上 | L 0.3cm@f74 / R 0.5cm@f74 | 0.113m@f380 / 0.113m@f124 | 0.082m@f173 / 0.074m@f172 | 1.00/1.00 |
| S11061（S11 动作06，Skeleton2） | 足最低点 2–6mm | L 0.2cm@f104 / R 0.6cm@f221 | 0.106m@f220 / 0.111m@f220 | 0.072m@f218 / 0.066m@f219 | 1.00/1.00 |
| S12011（S12 动作01，Skeleton3） | 最极端：右足足最低点 max 1mm | L 0.3cm@f453 / R 0.1cm@f185 | 0.100m@f432 / 0.101m@f94 | 0.061m@f396 / 0.051m@f184 | 1.00/1.00 |

## 逐 session 异常位置与情况

### S11053 —— 高速摆腿 + 极低趾间隙

- frame **237**：左足摆动峰值——BVH 踝 0.231m（23cm 高位）而左足足最低点仅 4.0cm；
- frame **224**：右足足最低点峰值 5.0cm（踝 0.24m 高位）；
- frame **64**：SMPL 侧右踝摆动峰值 0.185m。
- 情况：摆动脚速 ~4 m/s 的正常摆腿，趾尖却从未越过 bvh_h 的 6cm 迟滞线 →
  bvh_h/smpl_contact 双双全接触。BVH 与 SMPL 两源一致，非管线错误。

### S5091 —— 趾下垂摆动典型

- frame **316**：左踝摆到 0.188m 高位，但 SMPL 左足仍有 55 个足底顶点离地 <2cm
  （趾下垂贴地）；SMPL 左踝 0.146m；
- frame **366**：左足足最低点峰值 6.2cm（越过 6cm 线，但单帧未形成迟滞翻转）；
- frame **224**：右踝摆动峰值（BVH 0.189m / SMPL 0.128m）。

### S11032 —— 踝摆 25cm 而趾尖仅 4.8cm

- frame **327**：左足足最低点峰值 4.8cm（踝同时在高位）；
- frame **60**：右足足最低点峰值 5.4cm；SMPL 右踝 0.204m；
- frame **84**：BVH 左踝摆动峰值 0.252m。

### S11073 —— 对照组（唯一 bvh_h 出现离地帧）

- frame **65**：左足足最低点 6.5cm（越过 6cm 迟滞线 → bvh_h 判离地）；
- frame **44**：右足足最低点 6.3cm；
- frame **64**：双踝摆动峰值（BVH 0.231m / SMPL 0.179m）。

### S11011 / S10101 / S10103 / S11061 / S12011 —— 足最低点近乎全程贴地

- S11011 f483（L 0.7cm）/ f34（R 0.8cm）：踝摆 12–13cm 但足最低点不足 1cm；
- S10101 f65（L 0.3cm / R 0.5cm）、S10103 f74（同量级）：拖步式步态；
- S11061 f104（L 0.2cm）/ f221（R 0.6cm）；
- S12011 f453（L 0.3cm）/ f185（R 0.1cm）——右足足最低点全程最高仅 1mm。


文件树：

```text
motion_data/
├── README.md                              # 本说明
├── smpl_model/SMPL_NEUTRAL.pkl            # SMPL neutral 模板（235MB）
└── <sid>/
    ├── <sid>_SkeletonN.bvh                # 原始 BVH（cm，y-up，120Hz）
    └── motion_neutral_smpl.npz            # SMPL neutral 拟合（m，y-up）
```
