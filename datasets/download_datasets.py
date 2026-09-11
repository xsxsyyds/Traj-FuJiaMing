#!/usr/bin/env python3
"""下载并整理 ETH / UCY 行人轨迹数据集。

用法：
    python datasets/download_datasets.py            # 全部下载
    python datasets/download_datasets.py --no-video # 跳过视频，仅取轨迹数据

产物写入 datasets/eth/ 与 datasets/ucy/。已存在的文件会跳过（支持断点续传）。
UCY 的轨迹数据在压缩包中为嵌套 .rar，需要系统安装 WinRAR 或 7-Zip。
"""

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "_raw"
ETH_URL = "https://data.vision.ee.ethz.ch/cvl/aem/ewap_dataset_full.tgz"
UCY_URL = "https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data.zip"

UCY_SCENES = {
    "crowds_zara01": "zara01",
    "crowds_zara02": "zara02",
    "students003": "students03",
}


def fetch(url: str, dest: Path) -> Path:
    """下载 url 到 dest，已存在且非空则跳过。"""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[跳过] {dest.name} 已存在")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[下载] {url}")
    # curl 支持 -C - 断点续传，比 urllib 更适合大文件
    cmd = ["curl", "-L", "-C", "-", "--retry", "3", "--retry-delay", "2",
           "-o", str(dest), url]
    if shutil.which("curl"):
        subprocess.run(cmd, check=True)
    else:
        import urllib.request
        urllib.request.urlretrieve(url, dest)
    print(f"[完成] {dest.name}  {dest.stat().st_size / 1048576:.2f} MB")
    return dest


def find_unrar() -> str:
    """定位 WinRAR 或 7-Zip 的解压程序。"""
    candidates = [
        r"C:\Program Files\WinRAR\UnRAR.exe",
        r"C:\Program Files (x86)\WinRAR\UnRAR.exe",
        "unrar", "7z", "7za",
    ]
    for c in candidates:
        if shutil.which(c) or (os.path.sep in c and Path(c).exists()):
            return c
    raise RuntimeError(
        "未找到解压 .rar 的工具。请安装 WinRAR 或 7-Zip 后重试。"
    )


def extract_eth(dest_root: Path, keep_video: bool) -> None:
    tgz = fetch(ETH_URL, CACHE / "ewap_dataset_full.tgz")
    tmp = CACHE / "_eth"
    if not tmp.exists():
        print("[解压] ETH 压缩包")
        tmp.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tgz) as t:
            t.extractall(tmp)

    for seq in ("seq_eth", "seq_hotel"):
        src = tmp / "ewap_dataset" / seq
        dst = dest_root / "eth" / seq
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if f.suffix == ".avi" and not keep_video:
                continue
            shutil.copy2(f, dst / f.name)
        print(f"[组装] datasets/eth/{seq}")

    readme = tmp / "ewap_dataset" / "README.txt"
    if readme.exists():
        shutil.copy2(readme, dest_root / "eth" / "README.txt")


def extract_ucy(dest_root: Path, keep_video: bool) -> None:
    z = fetch(UCY_URL, CACHE / "crowd-data.zip")
    tmp = CACHE / "_ucy"
    if not tmp.exists():
        print("[解压] UCY 外层 zip")
        tmp.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(z) as f:
            f.extractall(tmp)

    data = tmp / "crowds" / "data"
    unrar = find_unrar()
    traj = CACHE / "_ucy_traj"
    traj.mkdir(parents=True, exist_ok=True)

    # 轨迹数据藏在嵌套 rar 中
    for rar in ("data_zara.rar", "data_university_students.rar"):
        print(f"[解压] {rar}")
        subprocess.run([unrar, "x", "-o+", "-idq", str(data / rar), str(traj) + os.sep],
                       check=True)

    for stem, scene in UCY_SCENES.items():
        dst = dest_root / "ucy" / scene
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(traj / f"{stem}.vsp", dst / f"{stem}.vsp")
        if keep_video:
            for ext in (".avi", ".jpg"):
                src = data / f"{stem}{ext}"
                if src.exists():
                    shutil.copy2(src, dst / src.name)
        # students003 的场景图命名不一致
        if not keep_video:
            jpg = data / ("students_003.jpg" if stem == "students003"
                          else f"{stem}.jpg")
            if jpg.exists():
                shutil.copy2(jpg, dst / jpg.name)
        print(f"[组装] datasets/ucy/{scene}")

    fmt = data / "crowd_file_format.txt"
    if fmt.exists():
        shutil.copy2(fmt, dest_root / "ucy" / "crowd_file_format.txt")


def main() -> int:
    ap = argparse.ArgumentParser(description="下载 ETH / UCY 行人轨迹数据集")
    ap.add_argument("--no-video", action="store_true",
                    help="不下载 avi 视频，仅取轨迹标注与场景图")
    args = ap.parse_args()
    keep_video = not args.no_video

    CACHE.mkdir(parents=True, exist_ok=True)
    extract_eth(ROOT, keep_video)
    extract_ucy(ROOT, keep_video)
    print("\n全部完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
