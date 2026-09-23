"""绘制单场景预测图：原始轨迹淡化（低不透明度 + 低饱和度），预测段加粗（满不透明度 + 满饱和度）。

协议与 evaluate_classroom.py 完全一致，保证画出来的就是「原代码」的结果：
  - 只把 8 帧历史喂给模型（不传真实未来），64 个邻居全部保留；
  - 单次均值 rollout，预测 12 帧；
  - primary 行人 = 场景第 0 列，ADE/FDE 以换算后坐标单位计（cm→m 仍是未确认假设）。

关于「第 0 列就是 primary」（已核对 trajnetplusplustools 源码，勿改）：
  Reader.track_rows_to_paths() 返回 [primary_path] + list(other_paths.values())，
  即主预测对象被强制放在下标 0；Reader.paths_to_xy() 用
  xy = np.full((frames, peds, 2), np.nan) 建数组，本数据集轨迹同步完整，
  故形状恒为 (20, 64, 2)。模型的 batch 约定同样是 index 0 = primary。

用法（环境就绪后）：
    python plot_prediction.py OUTPUT_BLOCK/circle_classroom/lstm_social_baseline.pkl \
        --split val --scene-id 0 --all-peds --label baseline

本脚本只读模型与数据、只写图片，不修改任何模型或数据集。
"""

from __future__ import annotations

import argparse
import colorsys
import json
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import to_rgb

import trajnetplusplustools

ROOT = Path(__file__).resolve().parent          # week03/Week3-TrajNet++/
WEEK = ROOT.parent                              # week03/
DEFAULT_OUT_DIR = WEEK / "results"              # week03/results/

OBS_LENGTH = 8
PRED_LENGTH = 12

# 配色：预测段满饱和，原始轨迹取其降饱和版本
PRED_COLOR = "#C00000"      # 深红，S = 1.0（满饱和）
CV_COLOR = "#1F4E79"        # 深蓝，仅在 --show-cv 时使用
PED_COLOR = "#B5B5B5"       # 其他行人，仅作背景
BOUNDARY_COLOR = "#3A3A3A"  # 历史/预测分界点


# --------------------------------------------------------------------------
# 样式
# --------------------------------------------------------------------------
def pick_font() -> str | None:
    """挑一个本机存在的 CJK 字体。图题是中文，缺字体会渲染成方框。"""
    prefer = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "PingFang SC",
        "SimSun",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in prefer:
        if name in available:
            return name
    return None


def apply_style() -> None:
    """图题中文、坐标与图例英文；负号用 ASCII 减号避免缺字形。"""
    family = pick_font()
    if family:
        plt.rcParams["font.family"] = family
        print(f"[style] font.family = {family}")
    else:
        print("[style] 警告：未找到任何 CJK 字体，中文图题会显示为方框")
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.edgecolor"] = "#3A3A3A"
    plt.rcParams["axes.labelcolor"] = "#3A3A3A"
    plt.rcParams["xtick.color"] = "#3A3A3A"
    plt.rcParams["ytick.color"] = "#3A3A3A"


def desaturate(color: str, sat: float) -> tuple[float, float, float]:
    """把颜色在 HSV 空间里降低饱和度，色相与明度不变。sat=1 表示不变。"""
    r, g, b = to_rgb(color)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return colorsys.hsv_to_rgb(h, min(1.0, max(0.0, s * sat)), v)


def ped_color(ped_id, primary_id=1):
    """行人配色：**按行人 ID 稳定取色**，同一个人在任何图里颜色都一致。

    primary 固定用预测色（深红）；其他人用黄金角在色环上散布，色相限制在
    [0.08, 0.92] 区间以避开红色（红色留给 primary）。
    """
    if int(ped_id) == int(primary_id):
        return PRED_COLOR
    lo, span, golden = 0.08, 0.84, 0.6180339887498949
    return colorsys.hsv_to_rgb(lo + ((int(ped_id) * golden) % span), 0.80, 0.72)


# --------------------------------------------------------------------------
# 单个场景的推理
# --------------------------------------------------------------------------
def infer_scene(model, paths):
    """按 evaluate_classroom.py 的协议做一次前向，返回 (xy, history, pred, cv, all_pred)。

    xy       : (20, N, 2) 全部行人的真实轨迹（第 0 列是 primary）
    pred     : (12, 2)    primary 的模型预测
    cv       : (12, 2)    primary 的匀速外推基线
    all_pred : (12, N, 2) **全部 N 个行人**的模型预测
    ped_ids  : 长度 N 的列表，每列对应的真实行人 ID（第 0 个是 primary）

    注意：模型本来就为场景里每个行人都输出预测（见 lstm.py forward 的
    pred_scene : Tensor [pred_length, num_tracks, 2]，docstring 是
    "Forecast the entire sequence"）。evaluate_classroom.py 只取 positions[-12:, 0]，
    是因为 TrajNet++ 官方指标只评 primary 行人，不是模型只预测了一个人。
    """
    xy = torch.tensor(trajnetplusplustools.Reader.paths_to_xy(paths), dtype=torch.float32)
    if xy.shape[0] != OBS_LENGTH + PRED_LENGTH:
        raise ValueError(f"场景帧数应为 {OBS_LENGTH + PRED_LENGTH}，实际 {xy.shape[0]}")
    history = xy[:OBS_LENGTH].clone()
    with torch.no_grad():
        _, positions = model(
            history,
            torch.zeros(xy.shape[1], 2),
            torch.tensor([0, xy.shape[1]]),
            n_predict=PRED_LENGTH,
        )
    all_pred = positions[-PRED_LENGTH:]          # (12, N, 2)
    pred = all_pred[:, 0]                        # primary
    last, prev = history[-1, 0], history[-2, 0]
    cv = last + torch.arange(1, PRED_LENGTH + 1)[:, None] * (last - prev)
    # 每列对应的真实行人 ID（第 0 列是 primary）—— 用于按 ID 稳定取色
    ped_ids = [int(path[0].pedestrian) for path in paths if path]
    return xy, history, pred, cv, all_pred, ped_ids


def ade_fde(pred: torch.Tensor, truth: torch.Tensor) -> tuple[float, float]:
    """平均/最终位移误差，与评价脚本同口径。"""
    d = torch.linalg.vector_norm(pred - truth, dim=-1)
    return float(d.mean()), float(d[-1])


# --------------------------------------------------------------------------
# 原始数据（仅用于 --panorama 的全景对照）
# --------------------------------------------------------------------------
def load_raw():
    """读原始 txt（换算为米），返回 (rows, frames, ids, scale)；读不到则 None。

    rows: {(frame, ped): np.array([x, y])}
    """
    txt = ROOT / "classroom/raw/circle-10m-64-1.txt"
    if not txt.is_file():
        return None
    scale = 0.01
    try:
        audit = json.loads((ROOT / "classroom/data_audit.json").read_text(encoding="utf-8"))
        scale = float(audit.get("coordinate_scale", scale))
    except Exception:
        pass
    rows = {}
    for line in txt.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        v = line.split()
        rows[(int(v[1]), int(v[0]))] = np.array([float(v[2]) * scale, float(v[3]) * scale])
    frames = sorted({k[0] for k in rows})
    ids = sorted({k[1] for k in rows})
    return rows, frames, ids, scale


def load_sample_map():
    """采样下标 -> 原始帧号（来自 data_audit.json）；读不到则 None。"""
    try:
        audit = json.loads((ROOT / "classroom/data_audit.json").read_text(encoding="utf-8"))
        return list(audit["sampled_to_original_frame"])
    except Exception:
        return None


def load_scene_meta(ndjson: Path):
    """直接解析 ndjson 的 scene 行，取 {sid: (start_idx, end_idx, primary_id)}。

    不依赖 trajnetplusplustools 内部属性名，最稳。
    """
    meta = {}
    for line in ndjson.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        sc = o.get("scene")
        if sc:
            meta[sc["id"]] = (sc["s"], sc["e"], sc["p"])
    return meta


# --------------------------------------------------------------------------
# 绘图
# --------------------------------------------------------------------------
def plot_scene(xy, history, pred, cv, *, sid, split, label, args, ax=None, save=True,
               all_pred=None, ped_ids=None):
    gt = xy[:, 0].numpy()                      # primary 的「全程」原始轨迹（20 点）
    hist = history[:, 0].numpy()
    pred_np = pred.numpy()
    cv_np = cv.numpy()
    truth = gt[OBS_LENGTH:]

    model_ade, model_fde = ade_fde(pred, xy[OBS_LENGTH:, 0])
    cv_ade, cv_fde = ade_fde(cv, xy[OBS_LENGTH:, 0])

    gt_color = args.gt_color or PRED_COLOR
    gt_color = tuple(desaturate(gt_color, args.gt_sat))
    pred_color = PRED_COLOR

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=tuple(args.figsize))
    else:
        fig = ax.figure

    n_ped = int(xy.shape[1])

    # ======================================================================
    # 模式 A：画全部行人的「原始轨迹 + 预测」（--all-preds），每人不同颜色
    # ======================================================================
    if args.all_preds and all_pred is not None:
        ap = all_pred.numpy()                  # (12, N, 2)
        prim_id = ped_ids[0] if ped_ids else 1
        if ped_ids is None:
            ped_ids = list(range(1, n_ped + 1))

        # 先画所有非 primary（原始淡化 → 预测加粗），primary 留到最后压顶
        order = [i for i in range(1, n_ped)] + [0]
        for i in order:
            c = ped_color(ped_ids[i], prim_id)
            is_primary = (ped_ids[i] == prim_id)
            tr_gt = xy[:, i].numpy()
            tr_pred = ap[:, i]
            lw_gt = args.pred_lw if is_primary else max(0.9, args.gt_lw)
            lw_pred = args.pred_lw if is_primary else max(1.5, args.pred_lw * 0.55)
            z = 4 if is_primary else 3

            # 原始轨迹：降饱和 + 降透明
            ax.plot(
                tr_gt[:, 0], tr_gt[:, 1], "-",
                color=tuple(desaturate(c, args.gt_sat)), lw=lw_gt,
                alpha=args.gt_alpha, zorder=z - 1, solid_capstyle="round",
            )
            # 预测段：满饱和 + 满透明 + 加粗
            line = np.vstack([history[-1, i].numpy(), tr_pred])
            ax.plot(
                line[:, 0], line[:, 1], "-", color=c, lw=lw_pred,
                alpha=1.0 if is_primary else args.pred_alpha,
                zorder=z, solid_capstyle="round",
            )

        # 图例用 primary 的颜色代表两类元素
        ax.plot([], [], "-", color=tuple(desaturate(pred_color, args.gt_sat)),
                lw=max(0.9, args.gt_lw), alpha=args.gt_alpha,
                label=f"Ground truth ({n_ped} peds)")
        ax.plot([], [], "-", color=pred_color, lw=args.pred_lw,
                label=f"Prediction ({n_ped} peds, coloured per ID)")
        ax.plot([], [], marker="o", ls="none", mfc="none", mec=BOUNDARY_COLOR,
                mew=1.2, ms=8, label=f"Obs/pred boundary (ID {prim_id})")

        if not args.no_metrics:
            txt = (f"primary ID {prim_id} ADE {model_ade:.3f} / FDE {model_fde:.3f}\n"
                   f"model predicts all {n_ped} pedestrians")
            if args.show_cv:
                txt += f"\nCV {cv_ade:.3f} / {cv_fde:.3f}"
            ax.text(
                0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left",
                fontsize=9, color="#3A3A3A",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#CCCCCC", alpha=0.85),
            )

    # ======================================================================
    # 模式 B：只画 primary（默认）
    # ======================================================================
    else:
        # --- 背景：其他行人的原始轨迹，只用来交代场景 ---
        if args.all_peds:
            for i in range(1, n_ped):
                t = xy[:, i].numpy()
                ax.plot(t[:, 0], t[:, 1], "-", color=PED_COLOR, lw=0.6, alpha=0.25, zorder=1)

        # --- 原始轨迹（全程）：低不透明度 + 低饱和度 ---
        ax.plot(
            gt[:, 0], gt[:, 1], "-", color=gt_color, lw=args.gt_lw,
            alpha=args.gt_alpha, zorder=2, label="Ground truth",
        )
        ax.plot(
            gt[:, 0], gt[:, 1], ".", color=gt_color, ms=3.5,
            alpha=args.gt_alpha, zorder=2, label="_nolegend_",
        )

        # --- 匀速基线（可选）---
        if args.show_cv:
            cv_line = np.vstack([hist[-1], cv_np])
            ax.plot(
                cv_line[:, 0], cv_line[:, 1], "--", color=CV_COLOR, lw=2.0,
                alpha=1.0, zorder=3, label="Constant velocity",
            )

        # --- 预测段：加粗 + 满不透明度 + 满饱和度 ---
        pred_line = np.vstack([hist[-1], pred_np])
        ax.plot(
            pred_line[:, 0], pred_line[:, 1], "-", color=pred_color, lw=args.pred_lw,
            alpha=1.0, zorder=4, label="Model prediction",
        )
        ax.plot(
            pred_np[:, 0], pred_np[:, 1], ".", color=pred_color, ms=5.0,
            alpha=1.0, zorder=4, label="_nolegend_",
        )

        # --- 指标标注（英文，图内标注约定用英文）---
        if not args.no_metrics:
            txt = f"ADE {model_ade:.3f} / FDE {model_fde:.3f}"
            if args.show_cv:
                txt += f"\nCV  {cv_ade:.3f} / {cv_fde:.3f}"
            ax.text(
                0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left",
                fontsize=9, color="#3A3A3A",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#CCCCCC", alpha=0.85),
            )

    # --- 历史/预测分界点（primary）---
    ax.plot(
        [hist[-1, 0]], [hist[-1, 1]], marker="o", mfc="none",
        mec=BOUNDARY_COLOR, mew=1.2, ms=8, zorder=5,
        label=("_nolegend_" if (args.all_preds and all_pred is not None) else "Obs/pred boundary"),
    )

    split_cn = "验证" if split == "val" else "测试"
    title = args.title or f"{label}：{split_cn}集场景 {sid}"
    ax.set(
        xlabel="x (scaled units)",
        ylabel="y (scaled units)",
        title=title,
    )
    ax.grid(True, color="#E5E5E5", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.axis("equal")

    if args.zoom:
        # --all-preds 时以「全部行人」为取景依据，否则只围 primary
        if args.all_preds and all_pred is not None:
            ref = xy.reshape(-1, 2).numpy()
        else:
            ref = gt
        cx = (ref[:, 0].min() + ref[:, 0].max()) / 2
        cy = (ref[:, 1].min() + ref[:, 1].max()) / 2
        half = max(
            (ref[:, 0].max() - ref[:, 0].min()) / 2,
            (ref[:, 1].max() - ref[:, 1].min()) / 2,
        ) + args.zoom
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)

    ax.legend(loc="lower right", fontsize=8, frameon=True, framealpha=0.9)

    out = Path(args.out) if args.out else (
        DEFAULT_OUT_DIR / f"pred_{label}_{split}_scene{sid}.png"
    )
    if not save:
        return out, (model_ade, model_fde), (cv_ade, cv_fde)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)

    if args.save_npz:
        npz = out.with_suffix(".npz")
        np.savez(
            npz,
            gt=gt, history=hist, pred=pred_np, cv=cv_np, truth=truth,
            scene_id=sid, split=split, label=label,
            model_ade=model_ade, model_fde=model_fde,
            cv_ade=cv_ade, cv_fde=cv_fde,
        )
        print(f"[save] {npz}")

    summary = {
        "scene_id": sid,
        "split": split,
        "label": label,
        "primary_index": 0,
        "pedestrians_in_scene": int(xy.shape[1]),
        "model": {"ADE": model_ade, "FDE": model_fde},
        "constant_velocity": {"ADE": cv_ade, "FDE": cv_fde},
        "units": "scaled coordinate units; metres only under unverified cm assumption",
        "protocol": "history only, all neighbours kept, one mean rollout",
        "figure": str(out),
    }
    print(f"[save] {out}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return out, (model_ade, model_fde), (cv_ade, cv_fde)


def plot_panorama(xy, history, pred, cv, *, sid, split, label, args, meta,
                  all_pred=None, ped_ids=None):
    """全景对照：左 = 该行人的完整行程（原始 txt，全部帧），右 = 本场景窗口与预测。

    直接回答「数据是 64 人圆环对趾，为什么预测图只看到一小段」：
    一个场景只有 20 个采样点，只是整段行程里的一小截。
    """
    raw = load_raw()
    smap = load_sample_map()
    if raw is None or smap is None or meta is None:
        print("[warn] 缺少原始 txt 或 data_audit.json，退化为普通单图", file=sys.stderr)
        return plot_scene(xy, history, pred, cv, sid=sid, split=split, label=label,
                          args=args, all_pred=all_pred, ped_ids=ped_ids)

    rows, frames, ids, _scale = raw
    s_idx, e_idx, pid = meta

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.3))

    # ---------------- 左：全景 ----------------
    axL = axes[0]
    first = np.array([rows[(frames[0], p)] for p in ids])
    center = first.mean(axis=0)
    radius = float(np.linalg.norm(first - center, axis=1).mean())
    th = np.linspace(0, 2 * np.pi, 500)
    axL.plot(center[0] + radius * np.cos(th), center[1] + radius * np.sin(th),
             "--", color="#3A3A3A", lw=1.1, zorder=1,
             label=f"起始圆 r≈{radius:.1f} 单位")

    # 每人不同颜色（--all-preds 时，按行人 ID 取色，与右图一致），否则统一浅灰
    for p in ids:
        tr = np.array([rows[(f, p)] for f in frames])
        if args.all_preds:
            axL.plot(tr[:, 0], tr[:, 1], "-", color=ped_color(p, pid),
                     lw=0.7, alpha=0.55, zorder=2)
        else:
            axL.plot(tr[:, 0], tr[:, 1], "-", color="#B5B5B5", lw=0.6, alpha=0.5, zorder=2)

    full = np.array([rows[(f, pid)] for f in frames])
    axL.plot(full[:, 0], full[:, 1], "-", color="#C00000", lw=2.0, alpha=0.65,
             zorder=3, label=f"ID {pid} 完整行程")

    # 本场景窗口（20 个采样点）—— 只有这么一小段
    wf = [smap[i] for i in range(s_idx, e_idx + 1)]
    win = np.array([rows[(f, pid)] for f in wf])
    axL.plot(win[:, 0], win[:, 1], "-", color="#C00000", lw=3.6, zorder=5,
             label=f"本场景窗口（{len(wf)} 原始帧）")

    axL.plot(full[0, 0], full[0, 1], "o", color="#1F4E79", ms=6, zorder=6, label="起点")
    axL.plot(full[-1, 0], full[-1, 1], "s", color="#2A9D8F", ms=6, zorder=6, label="对趾点")

    dur = (wf[-1] - wf[0] + 1) / 25.0
    total = (frames[-1] - frames[0] + 1) / 25.0
    axL.annotate(
        f"窗口 {len(wf)} 原始帧\n占全程 {dur / total * 100:.0f}%",
        xy=(float(win[:, 0].mean()), float(win[:, 1].mean())),
        xytext=(center[0] + radius * 0.40, center[1] + radius * 0.72),
        fontsize=8.5, color="#3A3A3A", ha="center",
        arrowprops=dict(arrowstyle="->", color="#3A3A3A", lw=1.0),
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#CCCCCC", alpha=0.9),
    )
    axL.set_title(f"全景：{len(ids)} 人圆环对趾，共 {len(frames)} 帧")
    axL.set_xlabel("x (scaled units)")
    axL.set_ylabel("y (scaled units)")

    # ---------------- 右：本场景窗口 ----------------
    plot_scene(xy, history, pred, cv, sid=sid, split=split, label=label,
               args=args, ax=axes[1], save=False, all_pred=all_pred, ped_ids=ped_ids)
    n_ped = int(xy.shape[1])
    right_title = (
        f"局部：全部 {n_ped} 人的窗口轨迹与预测\n（每人一色）"
        if args.all_preds
        else f"局部：本场景窗口与其预测\n（{len(history)} 点历史 + {len(pred)} 点预测）"
    )
    axes[1].set_title(right_title)

    for ax in axes:
        ax.axis("equal")
        ax.grid(True, color="#E5E5E5", lw=0.6)
        ax.set_axisbelow(True)
    axL.legend(loc="upper left", fontsize=7.5, frameon=True, framealpha=0.9)

    fig.tight_layout()
    out = Path(args.out) if args.out else (
        DEFAULT_OUT_DIR / f"pred_{label}_{split}_scene{sid}_panorama.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    print(f"[save] {out}")
    return out


# --------------------------------------------------------------------------
def main() -> int:
    # 中文提示在重定向到管道/文件时也保持可读（cmd 控制台本身走 UTF-8，不受影响）
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    p = argparse.ArgumentParser(
        description="画原版 Social LSTM 的单场景预测图：原始轨迹淡化、预测段加粗。"
    )
    p.add_argument("checkpoint", nargs="?", default=None,
                   help="你自己生成/信任的 checkpoint；用 --list-scenes 时可省略")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--scene-id", type=int, default=None,
                   help="要画的场景 id；默认取枚举到的第一个。用 --list-scenes 查看全部 id")
    p.add_argument("--label", default="baseline", help="图题里的方法名，也用于默认文件名")
    p.add_argument("--out", default=None, help="输出 PNG 路径；默认 week03/results/pred_<label>_<split>_scene<id>.png")
    p.add_argument("--title", default=None, help="覆盖图题（中文）")
    p.add_argument("--list-scenes", action="store_true", help="只列出场景 id 与人数，不画图")
    p.add_argument("--panorama", action="store_true",
                   help="左右对照：左=该行人完整行程（425 帧全景），右=本场景窗口（需原始 txt）")

    g = p.add_argument_group("样式")
    g.add_argument("--gt-alpha", type=float, default=0.45, help="原始轨迹不透明度（默认 0.45）")
    g.add_argument("--gt-sat", type=float, default=0.45, help="原始轨迹饱和度系数（默认 0.45）")
    g.add_argument("--gt-lw", type=float, default=1.3, help="原始轨迹线宽（默认 1.3）")
    g.add_argument("--gt-color", default=None, help=f"原始轨迹基色，默认随预测段 {PRED_COLOR}")
    g.add_argument("--pred-lw", type=float, default=3.4, help="预测段线宽（默认 3.4）")
    g.add_argument("--pred-alpha", type=float, default=1.0,
                   help="--all-preds 时非 primary 预测段的不透明度（默认 1.0）")
    g.add_argument("--zoom", type=float, default=None, metavar="PAD",
                   help="以 primary 轨迹为中心取景，四周留 PAD 个换算单位；不给则自动铺满全部已绘元素")
    g.add_argument("--figsize", type=float, nargs=2, default=[6.4, 6.4], metavar=("W", "H"))
    g.add_argument("--dpi", type=int, default=300, help="默认 300（交付只用高清 PNG）")
    g.add_argument("--all-peds", action="store_true", help="淡画其余行人的原始轨迹作为场景背景")
    g.add_argument("--all-preds", action="store_true",
                   help="画**全部 64 人**的原始轨迹与预测段，每人一种颜色（模型本就对所有人输出预测）")
    g.add_argument("--show-cv", action="store_true", help="额外画匀速外推基线")
    g.add_argument("--no-metrics", action="store_true", help="不在图上标注 ADE/FDE")
    g.add_argument("--save-npz", action="store_true", help="同时把 gt/pred/cv 存成 npz，便于之后重绘")

    args = p.parse_args()
    apply_style()

    # ---- 数据先就位：列场景不需要模型 ----
    ndjson = ROOT / f"DATA_BLOCK/circle_classroom/{args.split}/circle.ndjson"
    if not ndjson.is_file():
        print(f"[error] 找不到数据：{ndjson}", file=sys.stderr)
        return 2
    reader = trajnetplusplustools.Reader(str(ndjson), scene_type="paths")

    if args.list_scenes:
        n = 0
        for sid, paths in reader.scenes():
            n_ped = len(paths)
            n_frame = len(paths[0]) if paths else 0
            print(f"scene_id={sid}  pedestrians={n_ped}  frames={n_frame}")
            n += 1
        print(f"[info] {args.split} 共 {n} 个场景")
        return 0

    # ---- 画图必须有模型 ----
    if not args.checkpoint:
        print("[error] 没有给 checkpoint。画图需要模型，例如：", file=sys.stderr)
        print("  python plot_prediction.py OUTPUT_BLOCK/circle_classroom/"
              f"lstm_social_{args.label}.pkl --split {args.split}", file=sys.stderr)
        print("提示：先训练一次 -> python run_classroom.py --output " + args.label, file=sys.stderr)
        return 2
    ckpt = Path(args.checkpoint).resolve()
    if not ckpt.is_file():
        print(f"[error] 找不到 checkpoint：{ckpt}", file=sys.stderr)
        print("提示：先训练一次 -> python run_classroom.py --output " + args.label, file=sys.stderr)
        print("      产物默认落在 OUTPUT_BLOCK/circle_classroom/lstm_social_<output>.pkl", file=sys.stderr)
        return 2

    torch.set_num_threads(2)
    predictor = torch.load(ckpt, map_location="cpu")
    model = predictor.model.eval()

    target = args.scene_id
    scene_meta = load_scene_meta(ndjson) if args.panorama else {}
    for sid, paths in reader.scenes():
        if target is not None and sid != target:
            continue
        xy, history, pred, cv, all_pred, ped_ids = infer_scene(model, paths)
        if args.panorama:
            plot_panorama(xy, history, pred, cv, sid=sid, split=args.split,
                          label=args.label, args=args, meta=scene_meta.get(sid),
                          all_pred=all_pred, ped_ids=ped_ids)
        else:
            plot_scene(xy, history, pred, cv, sid=sid, split=args.split,
                       label=args.label, args=args, all_pred=all_pred, ped_ids=ped_ids)
        return 0

    print(f"[error] 没找到 scene_id={target}（数据里没有这个场景）", file=sys.stderr)
    print(f"提示：用 --list-scenes 查看 {args.split} 下可用的场景 id", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
