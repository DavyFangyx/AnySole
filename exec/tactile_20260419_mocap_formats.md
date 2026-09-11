# Tactile 20260419 MOCAP 导出文件格式与内部结构

调研日期：2026-09-09  
调研范围：`/data/lizhe/projects/Tactile/1_Data/20260419/mocap_ori_{bvh,c3d,cmr,trc}`  
任务分类：`raw_data` + `research`  
数据安全：全程只读，未修改 `1_Data/` 下任何文件。

## 1. 结论先读

这四类文件是同一次动捕 take 的不同表达，不是四次独立采集：

| 格式 | 核心内容 | 这批数据的实际粒度 | 是否保存真实 marker XYZ |
|---|---|---|---|
| BVH | 骨架树、关节静态 offset、根关节平移、每关节局部旋转 | 23 个骨架关节，120 Hz | 否；它是骨架求解结果 |
| TRC | 按帧保存带名称的 3D 点轨迹 | session 中每套 skeleton 有 53 个 marker，另有 Body_0/未命名点 | 是 |
| C3D | 标准二进制动捕容器，可同时容纳 3D point 和 analog | 本批只有 point，没有 analog | 是，但坐标轴约定与 TRC 导出不同 |
| CMR | CMTracker 的文本 record/replay 导出 | session 每帧 23 个骨架 body 的全局位置+四元数；标定 take 每帧 1 个 Body_0 | 否；它保存刚体/骨架 body pose |

用一句话区分：**TRC/C3D 主要是“光学点在哪里”，BVH/CMR 主要是“骨架解算后的人体怎样运动”。**

## 2. 目录和文件对应

20260419 共有 34 个 take：`S1011` 到 `S1113` 的 33 个 session，加上 1 个 `Take_calication` 标定 take。四种导出的 basename 集合完全一致，无缺失配对。

```text
mocap_ori_bvh/<take>/
  <take>_Skeleton1.bvh       # 33 个普通 session
  <take>_Video_44110129.avi  # MOCAP 系统预览视频

mocap_ori_bvh/Take_calication/
  Take_calication_Skeleton0.bvh
  Take_calication_Video_44110129.avi

mocap_ori_c3d/<take>.c3d
mocap_ori_cmr/<take>_record.cmr
mocap_ori_trc/<take>.trc
```

数量与大小：

| 目录 | 文件数 | 表观大小 |
|---|---:|---:|
| `mocap_ori_bvh` | 34 BVH + 34 AVI | 约 1.1 GiB，其中 BVH 合计 32,227,370 bytes，AVI 合计 1,059,665,388 bytes |
| `mocap_ori_c3d` | 34 | 约 84 MiB |
| `mocap_ori_cmr` | 34 | 约 93 MiB |
| `mocap_ori_trc` | 34 | 约 159 MiB |

## 3. BVH：骨架层级+局部运动通道

BVH（Biovision Hierarchy）是 ASCII 文本，分两段。

### 3.1 `HIERARCHY`

- `ROOT Hips`：根关节。
- `{ JOINT ... }`：嵌套树定义父子骨架。
- `OFFSET x y z`：子关节相对父关节的静态偏移，表示骨段长度/方向。
- `CHANNELS`：该关节在每个 motion frame 中占用的通道及顺序。
- `End Site`：头顶、手端、足尖等结束点，只有 offset，没有自己的运动通道。

S1011 的 root 通道为：

```text
CHANNELS 6 Xposition Yposition Zposition Yrotation Xrotation Zrotation
```

其余 22 个骨架关节各有 3 个 Euler 旋转通道：

```text
CHANNELS 3 Yrotation Xrotation Zrotation
```

因此每帧共 `6 + 22 x 3 = 72` 个数。这些数是“根关节平移+各关节局部 Euler 角”，不是 23 组现成的 XYZ。要得到 23 个关节的全局 XYZ，需要沿树执行 forward kinematics（FK）。

23 个骨架关节的顺序是：

```text
Hips, Spine, Spine1, Spine2, Spine3, Neck, Head,
LeftShoulder, LeftArm, LeftForeArm, LeftHand,
RightShoulder, RightArm, RightForeArm, RightHand,
LeftUpLeg, LeftLeg, LeftFoot, LeftToeBase,
RightUpLeg, RightLeg, RightFoot, RightToeBase
```

### 3.2 `MOTION`

S1011 的头部是：

```text
Frames: 2372
Frame Time: 0.00833333
```

即声明 2372 帧、120 Hz。第一行 motion 是零旋转的静态/初始骨架帧，之后才与 CMR/TRC 的采集帧对应。全部 34 个 take 都满足：

```text
BVH frames = TRC frames + 1
```

BVH 标准本身不写单位。根关节高度约 96、大腿/小腿 offset 约 40，且与 CMR 毫米坐标乘 0.1 后精确一致，可确认本批 BVH 的长度单位是厘米，旋转是度。

### 3.3 在本项目的意义

BVH 是 Stage 1 的 **MOCAP-derived 23-joint skeleton labels** 来源。它不是 53 个真实反光 marker。

## 4. TRC：按帧的真实 marker 全局 XYZ

TRC 是制表符分隔的 ASCII 文本。它的层次是：

```text
全局 metadata
  -> marker 名称列
    -> 每个 marker 的 X/Y/Z 三列
      -> 每帧一行数据
```

S1011 metadata：

```text
DataRate=120
CameraRate=120
NumFrames=2371
NumMarkers=122
Units=mm
OrigDataRate=120
OrigDataStartFrame=1
OrigNumFrames=2371
```

每个数据行的开头是：

```text
Frame#    Time    marker0_X marker0_Y marker0_Z ...
```

S1011 的 frame number 从 `881048` 开始，`Time` 从 `7342.066667 s` 开始；它们保留了动捕系统的全局帧索引/时间，而不是从 0 开始的 session-local 编号。

S1011 的 122 个点列为：

- `h_Skeleton0_*0..52`：53 个 marker；本 session 中为全 0，表示该 skeleton slot 未激活。
- `h_Skeleton1_*0..52`：53 个 marker；本 session 的有效人体 marker 轨迹。
- `Body_0_0..11`：12 个刚体 marker 列；在 S1011 中为全 0。
- `unlabel_7299..7302`：4 个未标注点。

普通 session 的 `NumMarkers` 为 118--129：固定基础是两套 53-marker skeleton slot 加 12 个 Body_0 列，差额来自数量不同的 `unlabel_*` 列。不能仅根据列存在就认定它有效，还要检查 XYZ 是否为非零有限值。

`Take_calication.trc` 不同：它有 844 帧、65 个列点，由 53 个 `h_Skeleton0_*` 列和 12 个 `Body_0_*` 列构成。其 skeleton 列全部无效，12 个 Body_0 marker 全部有效。这 12 个标定 marker 用于求 MOCAP raw 到棋盘格 world 的空间刚体变换。

## 5. C3D：同源 point 数据的标准二进制容器

C3D（Coordinate 3D）不是可直接阅读的文本表，而是标准二进制容器，内部包含：

1. 固定 header：point/analog 通道数、帧范围、采样率、data block 位置等。
2. parameter groups：`POINT`、`ANALOG`、`TIMECODE`、`MANUFACTURER`、`FORCE_PLATFORM` 等。
3. data blocks：按帧存储 point 和可选 analog 数据。

S1011 实际解析结果：

```text
POINT.USED       = 122
POINT.RATE       = 120 Hz
POINT.UNITS      = mm
POINT.FRAMES     = 2370
ANALOG.USED      = 0
MANUFACTURER     = ChingMU
SOFTWARE         = CMTracker
VERSION          = 2.0.0
```

`ezc3d` 将 point data 读成类似 `(4, 122, 2370)` 的数组，其中三个主分量是 XYZ，另有 residual/camera mask 等有效性元数据。本批 C3D 没有 analog 通道，因此没有力板/压力/肌电等同步模拟信号可用。

C3D 的 label 集与对应 TRC 一致。解析器把缺失点表示为 `NaN` 并提供 residual/validity 元数据，比 TRC 中单纯使用 `0,0,0` 更适合程序化有效性判断。

### 5.1 C3D 与 TRC 的两个实际差异

**帧数：**全部 34 个 take 都满足：

```text
C3D frames = TRC frames - 1
```

S1011 中，C3D 包含 TRC 的第一帧到倒数第二帧，TRC 的最后一帧 `883418` 没有进入 C3D。因此不能不检查长度就直接通过数组索引对齐。

**坐标轴：**S1011 同一 marker、同一帧的数值为：

```text
TRC xyz = ( 126.842445, 996.740540, 118.807098 )
C3D xyz = (-126.842445, 118.807098, 996.740540 )
```

这批数据的导出关系为：

```text
[X_c3d, Y_c3d, Z_c3d] = [-X_trc, Z_trc, Y_trc]
```

这是轴置换+一个符号反转，不是单位差异。

## 6. CMR：每帧 body 的全局位置+方向

CMR 在这批数据中是 CMTracker 的 ASCII record/replay 格式，不是像 C3D/BVH 那样的通用交换标准。文件自身没有单位、帧率、body 名称或四元数分量名的 schema header。

S1011 的结构是：

```text
totalFrame=2371
frameIndex=881048    bodyNum=23
bodySensor=450 info=x,y,z,qx,qy,qz,qw
bodySensor=451 info=x,y,z,qx,qy,qz,qw
...
bodySensor=472 info=x,y,z,qx,qy,qz,qw
frameIndex=881049    bodyNum=23
...
```

最后四个数在所检数据中范数均约等于 1，且与 BVH 旋转一致，因此可确认是单位四元数；按常见的 CMTracker 记录解释为 `(qx,qy,qz,qw)`。由于 CMR 本身没有分量标签，新解析器应把这一顺序做成显式配置并通过 BVH 反算验证，不应只凭扩展名假定。

S1011 的 `bodySensor=450..472` 依次对应 BVH 中上述 23 个关节。使用 BVH 第 1 帧（跳过 BVH 的静态第 0 帧）做 FK，23 个位置与 CMR 的对应位置全部吻合，最大误差约 `0.00015 cm`。坐标和单位关系为：

```text
[X_cmr, Y_cmr, Z_cmr] = 10 * [-X_bvh, Z_bvh, Y_bvh]
```

即 BVH 用 cm，CMR 位置用 mm；CMR 采用与这批 C3D 导出相同的轴排列。

`Take_calication_record.cmr` 每帧则是：

```text
frameIndex=446915 bodyNum=1
bodySensor=0 info=x,y,z,qx,qy,qz,qw
```

它保存的是 Body_0 刚体整体位姿，而 TRC/C3D 保存该刚体的 12 个独立 marker 全局坐标。

CMR 没有显式 FPS，但它的 `totalFrame`、`frameIndex` 与对应 TRC 完全对应；本批对应 TRC 是 120 Hz。

## 7. 四种文件如何相互对应

```text
光学相机观测反光点
            |
            +--> 点轨迹导出 --> TRC（文本）
            |                    \-> C3D（二进制标准容器）
            |
            +--> skeleton/rigid-body solver
                              +--> BVH（骨架树+局部通道）
                              \--> CMR（每个 body 的全局 pose）
```

具体到 S1011：

| 文件 | 帧数 | 起始索引 | 主体数据 |
|---|---:|---:|---|
| BVH | 2372 | session-local 0 | 1 个静态帧 + 2371 个运动帧，23-joint 局部通道 |
| TRC | 2371 | 881048 | 122 个 point 列，激活人体为 Skeleton1 的 53 markers |
| C3D | 2370 | 容器内 0 | 与 TRC 同源的 122 point labels，但少最后 1 帧 |
| CMR | 2371 | 881048 | 23 bodies 的全局 position + quaternion |

全部 34 个 take 都遵循同一帧数规律：

```text
BVH = TRC + 1
CMR = TRC
C3D = TRC - 1
```

这只是这批导出的实测事实，不是四种文件标准普遍规定。

## 8. 怎么选用

- 要训练/投影项目当前的 23 关节人体骨架：使用 BVH，做 FK 得到全局关节 XYZ。
- 要使用真实反光 marker，检查丢点或做 marker removal：使用 session TRC 中的 active `h_SkeletonN_*0..52`。
- 要做标定 Body_0：使用 `Take_calication.trc` 中 12 个 `Body_0_0..11` 的全局 XYZ，不要把它们当人体标签。
- 要一个通用、紧凑、可保留 point validity 和可选 analog 的交换容器：使用 C3D，但先显式统一轴约定，并处理这批数据末帧缺失。
- 要核对骨架 FK 的全局位置/方向或 Body_0 整体位姿：使用 CMR 作交叉验证。由于它缺少自描述 schema，不建议作为长期唯一 canonical 格式。

## 9. 最需要避免的误解

1. **23 joints 不等于 53 markers。** BVH/CMR 的 23 个 body/joint 是 skeleton solver 结果；TRC/C3D 的 53 个 active skeleton points 是真实 marker 轨迹。
2. **Body_0 的 12 markers 不是人体 marker。** 它们是标定刚体/棋盘格桥接点。
3. **同名 XYZ 不代表同轴约定。** 这批 C3D/CMR 与 TRC/BVH 之间存在 `[-X, Z, Y]` 轴变换。
4. **不要默认四个导出帧数相同。** 这批 BVH 有 1 个额外初始帧，C3D 比 TRC 少末尾 1 帧。
5. **TRC 列存在不等于点有效。** 未激活 skeleton slot 和普通 session 的 Body_0 可以整列为 0。

## 10. 验证方式

- 对四个目录做 basename 集合核对：34/34/34/34 一一匹配。
- 读取全部 34 个 BVH/TRC/CMR 头部和 C3D header/parameters，核对帧率、帧数、point 数和 analog 数。
- 用 `ezc3d` 解析 S1011 和 `Take_calication` 的 label、point shape 与有效性。
- 数值对比 S1011 的 TRC/C3D 首帧和尾帧，确认轴变换与 C3D 缺少 TRC 末帧。
- 用项目现有 BVH 解析/FK 实现计算 S1011 第 1 帧 23 joints，与 CMR 首帧 23 bodies 全部逐一对比。

## 11. 局限

- CMR 文本没有自描述 schema；其 position 单位、body 顺序和四元数语义是通过与 BVH/TRC 数值交叉验证得到，而非来自文件头声明。
- 本报告描述 20260419 这批 ChingMU CMTracker 2.0.0 导出的实际特性；帧数差和轴变换不应无条件推广到其他日期或其他软件导出。
