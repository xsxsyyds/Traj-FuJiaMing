# 数据集说明（ETH / UCY）

行人轨迹数据分析课程所用数据集，共 3 个场景组、5 个序列。

> **关于视频文件**：4 个 `.avi` 原始视频共约 222 MB，体积过大未纳入版本管理（已写入 `.gitignore`），仅保留在本地。仓库中的轨迹标注、场景平面图与说明文档共约 3 MB。需要视频时运行 `python datasets/download_datasets.py` 重新获取，或加 `--no-video` 只取数据。

## 目录结构

```
datasets/
├── eth/                        ETH Walking Pedestrians (EWAP)
│   ├── README.txt              官方说明（原文）
│   ├── seq_eth/
│   │   ├── obsmat.txt          轨迹标注（主数据）
│   │   ├── seq_eth.avi         原始视频
│   │   ├── map.png             场景障碍物平面图
│   │   ├── H.txt               单应矩阵（图像坐标 → 世界坐标）
│   │   ├── groups.txt          结伴行走的个体 ID 分组
│   │   ├── destinations.txt    假定的行人目的地
│   │   └── info.txt            序列信息
│   └── seq_hotel/              同 seq_eth 的文件构成
└── ucy/                        UCY Crowd Data
    ├── crowd_file_format.txt   官方 .vsp 格式说明（原文）
    ├── zara01/
    │   ├── crowds_zara01.vsp   轨迹标注（主数据）
    │   ├── crowds_zara01.avi   原始视频
    │   └── crowds_zara01.jpg   场景首帧图
    ├── zara02/
    └── students03/
```

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

| 序列 | 行人数 |
|---|---|
| zara01 | 148 |
| zara02 | 204 |
| students03 | 434 |

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

- 两套数据集的世界坐标系不一致：ETH 的位置已是米制的世界坐标，UCY 的 `.vsp` 是像素坐标，**跨数据集作图前需各自统一到米制**。
- 坐标轴方向不同，叠加平面图（`map.png` / `.jpg`）时注意垂直翻转。
- 数据集仅供研究教学使用，版权归原始作者所有。
