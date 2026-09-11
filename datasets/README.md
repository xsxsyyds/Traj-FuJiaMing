# 数据集说明（ETH / UCY）

行人轨迹数据分析课程所用数据集，共 3 个场景组、5 个序列。

> **关于视频文件**：4 个 `.avi` 原始视频共约 222 MB，体积过大未纳入版本管理（已写入 `.gitignore`），仅保留在本地。仓库中的轨迹标注、场景平面图与说明文档共约 3 MB。需要视频时运行 `python datasets/download_datasets.py` 重新获取，或加 `--no-video` 只取数据。

## 目录结构

```
datasets/
├── eth/                        ETH Walking Pedestrians (EWAP)
│   ├── README.txt              官方说明（原文）
│   ├── seq_eth/
│   │   ├── obsmat.txt          轨迹标注（主数据，米）
│   │   ├── seq_eth.avi         原始视频
│   │   ├── map.png             场景障碍物平面图
│   │   ├── H.txt               单应矩阵（图像坐标 → 世界坐标）
│   │   ├── groups.txt          结伴行走的个体 ID 分组
│   │   ├── destinations.txt    假定的行人目的地
│   │   └── info.txt            序列信息
│   └── seq_hotel/              同 seq_eth 的文件构成
└── ucy/                        UCY Crowd Data
    ├── crowd_file_format.txt   官方 .vsp 格式说明（原文）
    ├── build_ucy_metric.py     由 .vsp 生成米制版本
    ├── zara01/
    │   ├── zara01.txt          米制轨迹（分析用，主数据）
    │   ├── crowds_zara01.vsp   原始标注（像素坐标，溯源用）
    │   ├── crowds_zara01.avi   原始视频
    │   └── crowds_zara01.jpg   场景首帧图
    ├── zara02/
    └── students03/
```

> ⚠️ UCY 的 `.vsp` 是**图像像素坐标**，带透视压缩，不能直接计算速度、间距、密度。分析请用同目录下的 `<场景>.txt`（米制）。详见下文「UCY 坐标单位问题」。

## 来源

| 数据集 | 下载地址 | 压缩包 |
|---|---|---|
| ETH (EWAP) | https://data.vision.ee.ethz.ch/cvl/aem/ewap_dataset_full.tgz | 76.06 MB |
| UCY Crowd | https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data.zip | 151.52 MB |

UCY 原始包中的轨迹数据为嵌套 `.rar`（`data_zara.rar`、`data_university_students.rar`），已解出 `.vsp` 文件。

## 数据概览

### ETH（序列已缩放/标注，帧率 25 fps，每 10 帧标注一次即 2.5 fps）

| 序列 | 标注行数 | 行人数 | 帧范围 | 时长 |
|---|---|---|---|---|
| seq_eth | 8,908 | 360 | 780 – 12,381 | 约 7.7 min |
| seq_hotel | 6,544 | 390 | 1 – 18,061 | 约 12.0 min |

### UCY（每行人为一条样条曲线）

| 序列 | 行人数 | 米制文件点数 | 时长 |
|---|---|---|---|
| zara01 | 148 | 5,153 | 360 s |
| zara02 | 204 | 9,722 | 420 s |
| students03 | 434 | 17,953 | 216 s |

## 格式说明

### ETH `obsmat.txt`

每行 8 个字段，空格分隔：

```
[frame_number  pedestrian_ID  pos_x  pos_z  pos_y  v_x  v_z  v_y]
```

- `pos_z` 与 `v_z`（垂直于地面方向）未使用，恒为 0
- 位置与速度单位为**米**，由 `H.txt` 中的单应矩阵从图像坐标换算得到
- 行人 ID 在同一序列内唯一，跨序列不通用

### UCY `.vsp`

文件由若干条**样条曲线**组成，每条曲线描述一个人的运动：

```
<N>                     第一行：样条总数（即行人数）
<k1>                    第 1 条曲线的控制点数
x y frame gaze          控制点：平面坐标 x,y（像素）、帧号、视线方向
...                     （重复 k1 行）
<k2>                    第 2 条曲线的控制点数
...                     依次类推
```

注释以 `-` 开头，到行尾结束。详细格式见 `ucy/crowd_file_format.txt`。

`students003.vsp` 比另外两个文件多了**障碍物标注**：7 条线段障碍 + 33 个圆柱障碍（场景里的立柱），每个圆柱给出圆心、半径、类型、编号与存在时段。做 A5 瓶颈分析时可以直接用。

### UCY `<场景>.txt`（米制，分析用）

```
frame_id  ped_id  x[m]  y[m]
```

由 `datasets/build_ucy_metric.py` 生成，来源为学术界通行的 UCY 米制转换版，行人编号与原始 `.vsp` 一一对应（148 / 204 / 434）。

## UCY 坐标单位问题

**UCY 官方发布的 `.vsp` 是图像像素坐标，不是地面平面坐标。** 三条证据：

1. 坐标范围恰好等于视频像素边界。视频为 720×576，居中后应为 x ∈ [−360, 360]、y ∈ [−288, 288]；实测三个场景分别为 x ∈ [−368, 369]、[−366, 370]，高度吻合。
2. **透视压缩明显**：行人所在 y 位置与其像素速度显著负相关（Spearman rho = −0.33 ~ −0.48，p < 1e−4）。位置越靠上（离相机越远）移动越"慢"，这是透视投影的典型特征；若为地面平面坐标则不应存在该相关性。
3. 按像素直接当厘米换算，步行速度中位数仅 0.54 m/s，远低于正常步速 1.2–1.4 m/s。

转换后的米制版本中位步速约 1.16 m/s，符合实际。

**影响**：直接用 `.vsp` 作轨迹图，UCY 场景会被压成一条扁带；用其计算速度、间距、密度则完全失去物理意义。因此本仓库所有分析统一使用 `<场景>.txt`。

## 引用

使用这些数据请引用原始论文：

**ETH / EWAP**

> S. Pellegrini, A. Ess, K. Schindler, and L. Van Gool.
> You'll Never Walk Alone: Modeling Social Behavior for Multi-target Tracking.
> *IEEE International Conference on Computer Vision (ICCV)*, 2009.

**UCY**

> A. Lerner, Y. Chrysanthou, and D. Lischinski.
> Crowds by Example.
> *Computer Graphics Forum (Eurographics)*, 26(3):655–664, 2007.

## 使用注意

- **统一用米制**：ETH 用 `obsmat.txt`，UCY 用 `<场景>.txt`。不要用 `.vsp` 做定量分析。
- 两套数据集的原点、朝向、正负方向都不同，**不要跨数据集叠加作图**。
- ETH 的 `pos_z` / `v_z`（垂直地面方向）恒为 0，未使用。
- 视频帧率统一为 **25 fps**，帧号可直接除以 25 得到秒。ETH 标注每 10 帧一次（2.5 fps），UCY 米制版同样是每 10 帧一个采样点。
- 叠加平面图（`map.png` / `.jpg`）时注意垂直翻转。
- 数据集仅供研究教学使用，版权归原始作者所有。
