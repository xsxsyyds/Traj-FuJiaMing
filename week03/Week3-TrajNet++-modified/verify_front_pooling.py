# -*- coding: utf-8 -*-
"""验证前向半平面池化：① 是否只保留前方邻居；② 掩码是否随行人朝向旋转。

课堂说明第 4 节要求：「若研究前向视野，应先用历史速度定义局部朝向并验证旋转行为」。
本脚本用合成场景做三组对照（不需要数据、不需要训练，秒级出结果）：

  A. 上游 --front（type_='occupancy', front=True）
  B. 上游 --front 但把行人朝向转 90°
  C. 本次新增 front_social（前 180° 注意力池化）
  D. C 但把行人朝向转 90°

判据：把 1 个行人放在原点、4 个邻居分别放在 前(+x) / 后(-x) / 左(-y) / 右(+y)，
看哪几个格子被填上。若掩码跟随朝向，则 C→D 的存活邻居应整体旋转 90°；
若像上游那样固定在场景坐标轴上，则 A→B 的存活集合完全不变。

用法（在 Week3-TrajNet++-modified 目录下）：
    python verify_front_pooling.py
"""

import sys
from pathlib import Path

import torch

from trajnetbaselines.lstm.gridbased_pooling import GridBasedPooling

N = 4
CELL = 1.0
HIDDEN = 8
LATENT = 4

# 邻居相对被观察行人的位置（被观察行人位于原点）
NEIGHBOURS = {
    "front(+x)": (1.0, 0.0),
    "back(-x)": (-1.0, 0.0),
    "left(-y)": (0.0, -1.0),
    "right(+y)": (0.0, 1.0),
}


def rot90(v):
    """把向量逆时针旋转 90°：(x, y) -> (-y, x)"""
    return (-v[1], v[0])


def cell_index(rel):
    """按 occupancy 的同一公式把相对位置映射成格下标（pool_size=1）"""
    ox = rel[0] / CELL + N / 2
    oy = rel[1] / CELL + N / 2
    if not (0 <= ox < N and 0 <= oy < N):
        return None
    return int(ox) * N + int(oy)


def build_batch(heading):
    """构造 1 个场景、5 个行人：0 号是被观察者，1..4 是四个方向的邻居。

    heading: 被观察行人的朝向（单位向量）；邻居位置随之一起旋转。
    """
    heading = torch.tensor(heading, dtype=torch.float32)
    # 邻居相对位置：把模板位置按 heading 对应的旋转搬到朝向基准上
    # heading=(1,0) 时保持模板；heading=(0,1) 时整体逆时针转 90°
    if abs(heading[0] - 1.0) < 1e-6:
        rels = [NEIGHBOURS[k] for k in ("front(+x)", "back(-x)", "left(-y)", "right(+y)")]
    else:
        rels = [rot90(NEIGHBOURS[k]) for k in ("front(+x)", "back(-x)", "left(-y)", "right(+y)")]

    obs2 = torch.zeros(1, 1 + len(rels), 2)
    obs1 = torch.zeros(1, 1 + len(rels), 2)
    # 被观察行人：从原点沿 heading 走一步 => 速度 = heading
    obs1[0, 0] = torch.tensor([0.0, 0.0])
    obs2[0, 0] = heading
    # 邻居：位置 = 被观察者当前位置 + rel；给它一个平行于 heading 的速度
    for i, r in enumerate(rels, start=1):
        cur = obs2[0, 0] + torch.tensor(r, dtype=torch.float32)
        obs2[0, i] = cur
        obs1[0, i] = cur - heading
    return obs1, obs2, rels


def occupied_cells(grid):
    """从网格张量 [B*T, pooling_dim, n, n] 取出行人 0 的「被填格」下标集合。

    注意不要用 model(...)：那会再经过 embedding 返回交互向量，看不到网格。
    """
    g = grid[0]                      # 行人 0 的网格 [pooling_dim, N, N]
    mag = g.abs().sum(dim=0)         # 每个格的幅度
    occ = (mag > 1e-8).nonzero(as_tuple=False)
    if occ.numel() == 0:
        return set(), mag
    return {int(a) * N + int(b) for a, b in occ.tolist()}, mag


def make_upstream_front():
    """上游 --front：type_='occupancy' + front=True（occupancy 的格子填充逻辑
    与 social 共用同一个 occupancy() 方法，所以用它探针最干净：有邻居=1，空格=0）"""
    return GridBasedPooling(type_='occupancy', n=N, cell_side=CELL, front=True,
                            hidden_dim=HIDDEN, out_dim=8, embedding_arch='one_layer',
                            constant=0)


def make_new(front_half, ones_encoding=True):
    """本次新增的注意力池化；把编码层强制输出全 1，便于用"是否非零"判存活。"""
    p = GridBasedPooling(type_='front_social' if front_half else 'attn_social',
                         n=N, cell_side=CELL, hidden_dim=HIDDEN, latent_dim=LATENT,
                         out_dim=8, embedding_arch='one_layer', constant=0)
    if ones_encoding:
        torch.nn.init.constant_(p.hidden_dim_encoding.weight, 0.0)
        torch.nn.init.constant_(p.hidden_dim_encoding.bias, 1.0)
    p.eval()
    return p


def name_cells(cells, rels):
    """把格下标翻译回"哪个方向的邻居" """
    out = []
    for k, r in zip(("front", "back", "left", "right"), rels):
        idx = cell_index(r)
        if idx is None:
            out.append(f"{k}=界外")
        else:
            out.append(f"{k}={'✓' if idx in cells else '✗'}")
    return "  ".join(out)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    lines = []

    def say(s=""):
        lines.append(str(s))
        print(s)

    say("=" * 74)
    say("前向半平面池化验证（合成场景，1 个被观察者 + 4 个方向邻居）")
    say("=" * 74)
    say(f"网格 n={N}, cell_side={CELL}；邻居放在 ±1 个坐标单位处")
    say()

    results = {}

    # ---- A/B：上游 --front，朝向 +x 与 +y ----
    say("【上游 --front】它把网格平移到坐标轴一侧，不随朝向旋转")
    for tag, heading in (("A. 朝向 +x", (1.0, 0.0)), ("B. 朝向 +y（整体转 90°）", (0.0, 1.0))):
        model = make_upstream_front()
        obs1, obs2, rels = build_batch(heading)
        with torch.no_grad():
            grid = model.occupancies(obs1, obs2)
        cells, _ = occupied_cells(grid)
        results[tag] = cells
        say(f"  {tag:24s} 存活: {name_cells(cells, rels)}   (格下标 {sorted(cells)})")
    same = results["A. 朝向 +x"] == results["B. 朝向 +y（整体转 90°）"]
    say(f"  => 转 90° 后存活格的**下标集合**是否改变: "
        f"{'否 —— 两次都填同一个格，说明掩码固定在世界坐标轴、不随朝向旋转' if same else '是'}")
    say("     （两行标签看起来不同，只是因为邻居位置跟着一起转了，填的仍是同一格）")
    say()

    # ---- C/D：新增 front_social，朝向 +x 与 +y ----
    say("【本次新增 front_social】用历史速度定义朝向，掩码应随朝向旋转")
    for tag, heading in (("C. 朝向 +x", (1.0, 0.0)), ("D. 朝向 +y（整体转 90°）", (0.0, 1.0))):
        model = make_new(front_half=True)
        obs1, obs2, rels = build_batch(heading)
        hidden = torch.ones(1, obs2.size(1), HIDDEN)
        with torch.no_grad():
            grid = model.social_attention(hidden, obs1, obs2, front_half=True)
        cells, _ = occupied_cells(grid)
        results[tag] = (cells, rels)
        say(f"  {tag:24s} 存活: {name_cells(cells, rels)}   (格下标 {sorted(cells)})")

    # ---- E：新增 attn_social（不做方向筛选），应四面全收 ----
    say()
    say("【对照 attn_social】不做方向筛选，应四个方向全收")
    model = make_new(front_half=False)
    obs1, obs2, rels = build_batch((1.0, 0.0))
    hidden = torch.ones(1, obs2.size(1), HIDDEN)
    with torch.no_grad():
        grid = model.social_attention(hidden, obs1, obs2, front_half=False)
    cells_all, _ = occupied_cells(grid)
    say(f"  E. 朝向 +x（不筛方向） 存活: {name_cells(cells_all, rels)}")

    # ---- 判定 ----
    say()
    say("=" * 74)
    checks = []

    c_cells, c_rels = results["C. 朝向 +x"]
    fx = cell_index(c_rels[0])
    bx = cell_index(c_rels[1])
    lx = cell_index(c_rels[2])
    rx = cell_index(c_rels[3])
    checks.append(("C: 朝向 +x 时只保留正前方邻居", fx in c_cells))
    checks.append(("C: 朝向 +x 时排除正后方邻居", bx not in c_cells))
    checks.append(("C: 朝向 +x 时排除垂直两侧（投影=0，严格取 >0）",
                   (lx not in c_cells) and (rx not in c_cells)))

    d_cells, d_rels = results["D. 朝向 +y（整体转 90°）"]
    dfront = cell_index(d_rels[0])     # 旋转后位于 +y 的"前方"邻居
    dback = cell_index(d_rels[1])
    checks.append(("D: 朝向转 +y 后，前方邻居改为 +y 侧且仍存活", dfront in d_cells))
    checks.append(("D: 朝向转 +y 后，原 +x 邻居不再被当作前方", fx not in d_cells))
    checks.append(("D: 朝向转 +y 后，正后方邻居被排除", dback not in d_cells))
    checks.append(("A 与 B 存活集合相同 => 上游 --front 确实不随朝向旋转", same))
    checks.append(("E: attn_social 不做筛选时四个方向全存活", len(cells_all) == 4))

    ok = True
    for desc, passed in checks:
        say(f"  [{'PASS' if passed else 'FAIL'}] {desc}")
        ok &= bool(passed)

    say("=" * 74)
    say("结论：" + ("全部通过 —— 前向半平面掩码按历史速度定义、且随朝向旋转"
                  if ok else "存在未通过项，见上"))

    # 报告落到 results/（UTF-8），避免经 PowerShell 管道再次转码
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "verify_front_pooling.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
