# -*- coding: utf-8 -*-
"""按课堂说明汇总指标：**ADE、FDE、训练时间、模型参数量**。

（课堂说明第 2 节：「记录 ADE、FDE、训练时间、模型参数量，解释一个改善或失效案例」）

对每个 checkpoint 输出：
  - ADE / FDE：模型 与 匀速外推基线，可指定 val / test 两个划分
  - 训练时间：读同目录 `<ckpt>.classroom.json` 的 elapsed_seconds（含上游验证耗时）
  - 模型参数量：总参数与可训练参数，并单独给出池化模块的参数量
  - 实验配置：读 classroom.json 里的 arguments

评价协议与 `evaluate_classroom.py` 逐行一致，保证数字可比：
只传 8 帧历史、不传真实未来、64 个邻居全部保留、单次均值 rollout 预测 12 帧、
只评 primary 行人（TrajNet++ 官方口径）。

用法（在 week03/ 目录下）：
    python collect_metrics.py <ckpt1> [<ckpt2> ...] --splits val test
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

OBS_LENGTH = 8
PRED_LENGTH = 12

WEEK = Path(__file__).resolve().parent
# 两个代码副本里，-modified 是超集（新增了 attn_social / front_social 两个分支），
# 对 --type social 的类结构与原版完全一致，所以统一用它反序列化即可。
DEFAULT_PKG = WEEK / "Week3-TrajNet++-modified"


def load_predictor(ckpt: Path):
    """加载课堂包保存的 LSTMPredictor（内含 model）。"""
    obj = torch.load(ckpt, map_location="cpu")
    return obj.model


def count_params(model, pool=None):
    """参数量：总/可训练，以及池化模块单独的数量。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pool_total = 0
    if pool is not None:
        pool_total = sum(p.numel() for p in pool.parameters())
    return total, trainable, pool_total


def evaluate_split(model, ndjson: Path, obs_length=OBS_LENGTH, pred_length=PRED_LENGTH):
    """复刻 evaluate_classroom.py 的评价流程，返回 (模型误差, 匀速误差)。"""
    import trajnetplusplustools

    reader = trajnetplusplustools.Reader(str(ndjson), scene_type="paths")
    errs, cv_errs = [], []
    for _scene_id, paths in reader.scenes():
        xy = torch.tensor(trajnetplusplustools.Reader.paths_to_xy(paths), dtype=torch.float32)
        if xy.shape[0] != obs_length + pred_length:
            raise ValueError(f"{ndjson}: 场景帧数应为 {obs_length + pred_length}，实际 {xy.shape[0]}")
        history = xy[:obs_length].clone()
        with torch.no_grad():
            _, positions = model(history, torch.zeros(xy.shape[1], 2),
                                 torch.tensor([0, xy.shape[1]]), n_predict=pred_length)
        pred = positions[-pred_length:, 0]
        truth = xy[obs_length:, 0]
        last, prev = history[-1, 0], history[-2, 0]
        cv = last + torch.arange(1, pred_length + 1)[:, None] * (last - prev)
        errs.append(torch.linalg.vector_norm(pred - truth, dim=-1).numpy())
        cv_errs.append(torch.linalg.vector_norm(cv - truth, dim=-1).numpy())

    e = np.stack(errs)
    c = np.stack(cv_errs)
    if not np.isfinite(e).all():
        raise ValueError(f"{ndjson}: 模型预测出现非有限值")
    return ({"ADE": float(e.mean()), "FDE": float(e[:, -1].mean()), "scenes": int(e.shape[0])},
            {"ADE": float(c.mean()), "FDE": float(c[:, -1].mean())})


def label_of(ckpt: Path):
    """从 lstm_<type>_<output>.pkl 解析出可读标签。"""
    name = ckpt.name
    if name.startswith("lstm_") and name.endswith(".pkl"):
        body = name[len("lstm_"):-len(".pkl")]
        if "_" in body:
            typ, out = body.split("_", 1)
            return typ, out
    return "unknown", ckpt.stem


def collect(ckpt: Path, splits, pkg: Path):
    model = load_predictor(ckpt)
    model.eval()
    pool = getattr(model, "pool", None)
    total, trainable, pool_total = count_params(model, pool)
    typ, out = label_of(ckpt)

    rec = {
        "checkpoint": str(ckpt),
        "label": f"{typ}/{out}",
        "type": typ,
        "output_name": out,
        "params_total": total,
        "params_trainable": trainable,
        "params_pool": pool_total,
        "training": None,
        "metrics": {},
    }

    ## 训练时间与实验配置
    side = Path(str(ckpt) + ".classroom.json")
    if side.is_file():
        meta = json.loads(side.read_text(encoding="utf-8"))
        rec["training"] = {
            "elapsed_seconds": meta.get("elapsed_seconds"),
            "torch": meta.get("torch"),
            "numpy": meta.get("numpy"),
            "device": meta.get("device"),
            "threads": meta.get("threads"),
        }
        args = meta.get("arguments") or []
        cfg = {}
        for i, tok in enumerate(args[:-1]):
            if tok.startswith("--"):
                cfg[tok.lstrip("-")] = args[i + 1]
        rec["config"] = cfg

    ## 逐划分评价
    for split in splits:
        ndjson = pkg / f"DATA_BLOCK/circle_classroom/{split}/circle.ndjson"
        if not ndjson.is_file():
            rec["metrics"][split] = {"error": f"缺少数据 {ndjson}"}
            continue
        m, cv = evaluate_split(model, ndjson)
        rec["metrics"][split] = {"model": m, "constant_velocity": cv}
    return rec


def fmt(v, nd=4):
    return "n/a" if v is None else f"{v:.{nd}f}"


def to_markdown(recs, splits):
    lines = []
    lines.append("# week03 指标汇总（课堂说明要求：ADE、FDE、训练时间、模型参数量）\n")
    lines.append("评价协议与 `evaluate_classroom.py` 一致：只传 8 帧历史、不传真实未来、"
                 "64 人全部递推、单次均值 rollout、只评 primary 行人。\n")

    for split in splits:
        lines.append(f"\n## {split} 划分\n")
        head = "| 实验 | ADE | FDE | 匀速 ADE | 匀速 FDE | 参数量 | 训练时间 (s) |"
        sep = "|---|---|---|---|---|---|---|"
        lines.append(head)
        lines.append(sep)
        for r in recs:
            m = r["metrics"].get(split, {})
            if "model" not in m:
                lines.append(f"| {r['label']} | {m.get('error', 'n/a')} | - | - | - | "
                             f"{r['params_total']} | {fmt(r['training']['elapsed_seconds'], 1) if r['training'] else 'n/a'} |")
                continue
            mo, cv = m["model"], m["constant_velocity"]
            tt = fmt(r["training"]["elapsed_seconds"], 1) if r["training"] else "n/a"
            lines.append(
                f"| {r['label']} | {fmt(mo['ADE'])} | {fmt(mo['FDE'])} | "
                f"{fmt(cv['ADE'])} | {fmt(cv['FDE'])} | {r['params_total']} | {tt} |"
            )

    lines.append("\n## 参数量明细\n")
    lines.append("| 实验 | 总参数 | 可训练 | 其中池化模块 |")
    lines.append("|---|---|---|---|")
    for r in recs:
        lines.append(f"| {r['label']} | {r['params_total']} | {r['params_trainable']} | {r['params_pool']} |")

    if len(recs) >= 2:
        base = recs[0]
        lines.append(f"\n## 相对基线 `{base['label']}` 的差值\n")
        head = "| 指标 | 基线 | " + " | ".join(f"{r['label']}" for r in recs[1:]) + " |"
        lines.append(head)
        lines.append("|---" * (len(recs) + 1) + "|")
        for split in splits:
            mb = base["metrics"].get(split, {}).get("model")
            if not mb:
                continue
            for k in ("ADE", "FDE"):
                cells = []
                for r in recs[1:]:
                    mr = r["metrics"].get(split, {}).get("model")
                    cells.append(f"{fmt(mr[k])} ({mr[k] - mb[k]:+.4f})" if mr else "n/a")
                lines.append(f"| {split} {k} | {fmt(mb[k])} | " + " | ".join(cells) + " |")
        cells = []
        for r in recs[1:]:
            cells.append(f"{r['params_total']} ({r['params_total'] - base['params_total']:+d})")
        lines.append(f"| 参数量 | {base['params_total']} | " + " | ".join(cells) + " |")
        tt0 = base["training"]["elapsed_seconds"] if base["training"] else None
        cells = []
        for r in recs[1:]:
            t = r["training"]["elapsed_seconds"] if r["training"] else None
            cells.append(f"{fmt(t, 1)} ({t - tt0:+.1f})" if (t and tt0) else "n/a")
        lines.append(f"| 训练时间 (s) | {fmt(tt0, 1)} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    p = argparse.ArgumentParser(description="汇总 ADE / FDE / 训练时间 / 模型参数量")
    p.add_argument("checkpoints", nargs="+", help="一个或多个 checkpoint 路径")
    p.add_argument("--splits", nargs="+", default=["val", "test"], choices=["val", "test"])
    p.add_argument("--pkg", default=str(DEFAULT_PKG),
                   help="代码包目录（用于定位 DATA_BLOCK 并提供 trajnetbaselines 类定义）")
    p.add_argument("--out-md", default=str(WEEK / "results/metrics_comparison.md"))
    p.add_argument("--out-json", default=str(WEEK / "results/metrics_comparison.json"))
    args = p.parse_args()

    pkg = Path(args.pkg).resolve()
    if not (pkg / "trajnetbaselines").is_dir():
        print(f"[error] 不是有效的代码包目录：{pkg}", file=sys.stderr)
        return 2
    ## 让 pickle 能找到 trajnetbaselines 里的类
    sys.path.insert(0, str(pkg))

    torch.set_num_threads(2)

    recs = []
    for c in args.checkpoints:
        ckpt = Path(c).resolve()
        if not ckpt.is_file():
            print(f"[error] 找不到 checkpoint：{ckpt}", file=sys.stderr)
            return 2
        rec = collect(ckpt, args.splits, pkg)
        recs.append(rec)
        print(f"[ok] {rec['label']}  参数量={rec['params_total']}  "
              f"训练={fmt(rec['training']['elapsed_seconds'], 1) if rec['training'] else 'n/a'}s")
        for split in args.splits:
            m = rec["metrics"].get(split, {})
            if "model" in m:
                print(f"     {split}: ADE {fmt(m['model']['ADE'])} / FDE {fmt(m['model']['FDE'])}"
                      f"   (匀速 {fmt(m['constant_velocity']['ADE'])} / {fmt(m['constant_velocity']['FDE'])})")

    md = to_markdown(recs, args.splits)
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")
    Path(args.out_json).write_text(json.dumps(recs, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[save] {out_md}")
    print(f"[save] {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
