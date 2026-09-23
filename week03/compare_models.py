# -*- coding: utf-8 -*-
"""固定场景对比图：把多个模型在**同一个场景**上的预测并排画出来。

课堂说明第 5 节要求提交「一个固定场景对比图」。本脚本读入若干 checkpoint，
对同一个场景各跑一次前向，画成 1×N 面板：

  - 灰色淡化 = 原始轨迹（所有面板完全一致，作为共同参照）
  - 彩色加粗 = 该模型的预测段
  - 左上角标注该模型在这个场景上的 ADE / FDE

复用 `Week3-TrajNet++-modified/plot_prediction.py` 里的样式与推理函数，保证
口径与单图模式完全一致（只传 8 帧历史、64 人全递推、单次均值 rollout）。

用法（在 week03/ 目录下）：
    python compare_models.py \
        --model baseline=results/lstm_social_baseline.pkl \
        --model attn=results/lstm_attn_social_baseline.pkl \
        --model front=results/lstm_front_social_baseline.pkl \
        --split val --scene-id 56 --zoom 3
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

WEEK = Path(__file__).resolve().parent
DEFAULT_PKG = WEEK / "Week3-TrajNet++-modified"

# 每个模型的预测段颜色（满饱和）；原始轨迹统一用中性灰，保证各面板只差在预测上
MODEL_COLORS = ["#C00000", "#1F4E79", "#2A9D8F", "#8E44AD", "#D68910"]
GT_NEUTRAL = "#9AA5B1"

OBS_LENGTH = 8
PRED_LENGTH = 12


def parse_model(spec: str):
    """'label=path' -> (label, Path)"""
    if "=" not in spec:
        p = Path(spec)
        return p.stem, p
    label, path = spec.split("=", 1)
    return label.strip(), Path(path.strip())


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="多个模型在同一场景上的预测对比图")
    ap.add_argument("--model", action="append", required=True, metavar="LABEL=PATH",
                    help="可重复；至少给两个")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--scene-id", type=int, required=True)
    ap.add_argument("--zoom", type=float, default=3.0,
                    help="以 primary 轨迹为中心取景，四周留多少坐标单位")
    ap.add_argument("--gt-alpha", type=float, default=0.55,
                    help="原始轨迹不透明度（对比图里略微调高，否则压在粗预测线下看不见）")
    ap.add_argument("--pred-lw", type=float, default=3.4)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--pkg", default=str(DEFAULT_PKG))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pkg = Path(args.pkg).resolve()
    if not (pkg / "trajnetbaselines").is_dir():
        print(f"[error] 不是有效的代码包目录：{pkg}", file=sys.stderr)
        return 2
    sys.path.insert(0, str(pkg))

    import trajnetplusplustools
    import matplotlib.pyplot as plt
    import plot_prediction as pp          # 复用样式与推理

    pp.apply_style()
    torch.set_num_threads(2)

    models = [parse_model(s) for s in args.model]
    if len(models) < 2:
        print("[error] 至少给两个 --model 才谈得上对比", file=sys.stderr)
        return 2

    ndjson = pkg / f"DATA_BLOCK/circle_classroom/{args.split}/circle.ndjson"
    if not ndjson.is_file():
        print(f"[error] 找不到数据：{ndjson}", file=sys.stderr)
        return 2
    reader = trajnetplusplustools.Reader(str(ndjson), scene_type="paths")

    ## 取出目标场景的 paths（只拿一次，保证所有模型看到同一份输入）
    target_paths = None
    for sid, paths in reader.scenes():
        if sid == args.scene_id:
            target_paths = paths
            break
    if target_paths is None:
        print(f"[error] 数据里没有 scene_id={args.scene_id}", file=sys.stderr)
        return 1

    ## 逐模型推理
    runs = []
    for (label, ckpt), color in zip(models, MODEL_COLORS * 10):
        ckpt = ckpt.resolve()
        if not ckpt.is_file():
            print(f"[error] 找不到 checkpoint：{ckpt}", file=sys.stderr)
            return 2
        model = torch.load(ckpt, map_location="cpu").model.eval()
        xy, history, pred, cv, _all_pred, ped_ids = pp.infer_scene(model, target_paths)
        ade, fde = pp.ade_fde(pred, xy[OBS_LENGTH:, 0])
        cv_ade, cv_fde = pp.ade_fde(cv, xy[OBS_LENGTH:, 0])
        n_par = sum(p.numel() for p in model.parameters())
        runs.append(dict(label=label, color=color, xy=xy, history=history, pred=pred,
                         cv=cv, ade=ade, fde=fde, cv_ade=cv_ade, cv_fde=cv_fde,
                         params=n_par, ckpt=str(ckpt)))
        print(f"[ok] {label:12s} ADE {ade:.4f}  FDE {fde:.4f}  参数量 {n_par}")

    ## 出图：1×N 面板，同一取景
    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(6.0 * n, 6.0), squeeze=False)
    axes = axes[0]

    ## 统一取景：只以 primary 的原始轨迹 + 各模型的预测段为参照，
    ## 不要把全部行人算进去（否则 10 m 圆盘会把视野撑开、主体缩成一个点）
    gt = runs[0]["xy"][:, 0].numpy()
    ref = [gt]
    for r in runs:
        ref.append(np.vstack([r["history"][-1, 0].numpy(), r["pred"].numpy()]))
    ref = np.vstack(ref)
    cx = (ref[:, 0].min() + ref[:, 0].max()) / 2
    cy = (ref[:, 1].min() + ref[:, 1].max()) / 2
    half = max(ref[:, 0].max() - ref[:, 0].min(),
               ref[:, 1].max() - ref[:, 1].min()) / 2 + args.zoom

    for ax, r in zip(axes, runs):
        xy = r["xy"]
        gt_line = xy[:, 0].numpy()
        hist = r["history"][:, 0].numpy()
        pred = r["pred"].numpy()

        # 其他行人：极淡，交代场景
        for i in range(1, xy.shape[1]):
            t = xy[:, i].numpy()
            ax.plot(t[:, 0], t[:, 1], "-", color="#DDDDDD", lw=0.5, alpha=0.7, zorder=1)

        # 原始轨迹（全程，中性灰淡化）
        ax.plot(gt_line[:, 0], gt_line[:, 1], "-", color=GT_NEUTRAL,
                lw=1.4, alpha=args.gt_alpha, zorder=3, label="Ground truth")
        ax.plot(gt_line[:, 0], gt_line[:, 1], ".", color=GT_NEUTRAL,
                ms=3.2, alpha=args.gt_alpha, zorder=3)

        # 预测段：加粗满饱和
        line = np.vstack([hist[-1], pred])
        ax.plot(line[:, 0], line[:, 1], "-", color=r["color"], lw=args.pred_lw,
                alpha=1.0, zorder=5, solid_capstyle="round", label="Prediction")
        ax.plot(pred[:, 0], pred[:, 1], ".", color=r["color"], ms=4.5, alpha=1.0, zorder=5)

        # 分界点
        ax.plot([hist[-1, 0]], [hist[-1, 1]], marker="o", mfc="none",
                mec="#3A3A3A", mew=1.2, ms=8, zorder=6, label="Obs/pred boundary")

        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_aspect("equal")
        ax.grid(True, color="#E5E5E5", lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("x (scaled units)")
        if ax is axes[0]:
            ax.set_ylabel("y (scaled units)")

        ax.set_title(f"{r['label']}：ADE {r['ade']:.3f} / FDE {r['fde']:.3f}")
        ax.text(0.02, 0.02,
                f"params {r['params']:,}\nCV {r['cv_ade']:.3f} / {r['cv_fde']:.3f}",
                transform=ax.transAxes, va="bottom", ha="left", fontsize=8,
                color="#3A3A3A",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", alpha=0.85))

    axes[0].legend(loc="upper left", fontsize=7.5, frameon=True, framealpha=0.9)
    fig.suptitle(f"固定场景对比：{args.split} 场景 {args.scene_id}"
                 f"（raw trajectory vs. 各模型预测段）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = Path(args.out) if args.out else (
        WEEK / "results" / f"compare_{args.split}_scene{args.scene_id}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    print(f"[save] {out}")

    side = out.with_suffix(".json")
    side.write_text(json.dumps([
        {k: v for k, v in r.items() if k not in ("xy", "history", "pred", "cv")}
        for r in runs], indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[save] {side}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
