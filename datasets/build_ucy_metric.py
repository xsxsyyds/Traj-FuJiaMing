#!/usr/bin/env python3
"""把 UCY 原始标注（.vsp，图像像素坐标）转换为米制轨迹文件。

为什么需要这一步
----------------
UCY 官方发布的 `.vsp` 是**图像像素坐标**，带强烈透视压缩（远处行人的像素
速度明显更小，Spearman rho ≈ -0.33 ~ -0.48）。直接用它算速度、间距、密度
会得到没有物理意义的数值。

学术界通行的转换版（zara01.txt / zara02.txt / students03.txt）已通过单应
变换落到地面平面，单位为米，且行人编号与原始 `.vsp` 一一对应（148 / 204 /
434）。本脚本从公开的 Trajectron++ 仓库取回该转换版，合并 train/val 两个
时间分段，还原为完整序列。

输出格式（每行）
----------------
    frame_id  ped_id  x[m]  y[m]

用法
----
    python datasets/build_ucy_metric.py
"""

import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UCY = ROOT / "ucy"
CACHE = ROOT / "_raw" / "ucy_metric"

BASE = ("https://raw.githubusercontent.com/StanfordASL/"
        "Trajectron-plus-plus/master/experiments/pedestrians/raw/eth")

# 转换版文件名 -> 课程使用的场景名
SCENES = {
    "crowds_zara01": "zara01",
    "crowds_zara02": "zara02",
    "students003": "students03",
}

FPS = 25.0


def fetch(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[下载] {dest.name}")
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception:
        subprocess.run(["curl", "-sL", "-o", str(dest), url], check=True)
    return dest


def read_txt(p: Path):
    rows = []
    for line in p.read_text().splitlines():
        s = line.split()
        if len(s) >= 4:
            rows.append((float(s[0]), int(float(s[1])), float(s[2]), float(s[3])))
    return rows


def build(stem: str, scene: str) -> None:
    parts = []
    for split in ("train", "val"):
        p = fetch(f"{BASE}/{split}/{stem}_{split}.txt",
                  CACHE / f"{stem}_{split}.txt")
        parts.extend(read_txt(p))

    # 按 行人ID -> 帧号 排序，保证每人的轨迹按时间连续
    parts.sort(key=lambda r: (r[1], r[0]))

    out_dir = UCY / scene
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{scene}.txt"

    n_ped = len({r[1] for r in parts})
    f0 = min(r[0] for r in parts)
    f1 = max(r[0] for r in parts)

    with out.open("w") as f:
        for fr, pid, x, y in parts:
            f.write(f"{fr:.0f} {pid} {x:.6f} {y:.6f}\n")

    print(f"[写出] {out.relative_to(ROOT.parent)}  "
          f"{len(parts)} 点  行人 {n_ped}  帧 {f0:.0f}-{f1:.0f} "
          f"({(f1 - f0) / FPS:.0f}s)")


def main() -> int:
    print("UCY 原始标注为图像像素坐标，本脚本生成米制版本用于分析。\n")
    for stem, scene in SCENES.items():
        build(stem, scene)
    print("\n完成。原始 .vsp 仍保留在同目录下作为溯源依据。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
