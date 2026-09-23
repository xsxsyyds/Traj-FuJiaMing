# Week3-TrajNet++-modified —— 池化层修改版

本目录是课程下发的 TrajNet++ 课堂包**加上本次池化修改**后的完整代码副本。
原版保留在 `../Week3-TrajNet++/`，**未被改动**（其 `trajnetbaselines/` 与仓库 HEAD 一致）。

## 相对原版改了什么

只有两个文件被修改，另有三个新增脚本：

| 文件 | 状态 | 内容 |
|---|---|---|
| `trajnetbaselines/lstm/gridbased_pooling.py` | **修改** | 新增 `social_attention()` 方法与 `attn_social` / `front_social` 两个池化类型 |
| `trajnetbaselines/lstm/trainer.py` | **修改** | 把上述两个类型加入 `--type` 的 `choices` |
| `plot_prediction.py` | 新增 | 画单场景预测图（原始轨迹淡化、预测段加粗）；`--all-preds` 可画全部 64 人 |
| `verify_front_pooling.py` | 新增 | 合成场景验证前向半平面掩码及其旋转行为（8 项断言） |

其余文件与原版**逐字节一致**（`trajnetbaselines/` 下 32 个 `.py` 已用 sha256 逐个比对确认）。
完整补丁见 `../results/pooling_attention.patch`（+95 / −2 行）。

## 池化修改的逻辑

新增的 `--type` 有两个，用来把「注意力」和「前向掩码」两件事拆开看：

- **`attn_social`**：注意力加权池化，**全部方向**的邻居都参与。
- **`front_social`**：同上，但**只保留前 180° 半平面**内的邻居。

与上游 `social()` 的三点差别：

1. **朝向**：用历史速度 `v_i = obs2_i − obs1_i` 定义第 i 个行人自己的朝向
   （`obs1/obs2` 是相邻帧位置，与 `lstm.py` 里算嵌入用的 `curr_velocity` 是同一个量）。
   `front_social` 只保留满足 `dot(pos_j − pos_i, v_i) > 0` 的邻居 j；
   速度近乎 0 时朝向无定义，此时不筛方向，避免把邻居全丢掉。
2. **注意力权重**：邻居特征入格前先乘权重。打分
   `score_ij = ⟨W_q h_i, W_k e_j⟩ / √d`（query 来自行人自身隐状态，key 来自邻居压缩隐状态），
   **只在前向邻居上做 softmax**；没有任何前向邻居时权重全为 0。
3. **同格累加**：同格多邻居用 `scatter_add_` **累加**，替掉上游
   `occ[arange, oi] = other_values` 的「后写覆盖先写」。

> ⚠️ **一个必须知道的坑**：`generate_pooling_inputs` 用 `fill_(float('nan'))` 做 padding
> （`lstm.py` L31-34），上游 `social()` 是在编码**前**用 `nan_to_num` 清掉的。
> 新代码最初直接用原始 `hidden_state` 算 query，NaN 经 softmax 污染了
> `hidden_dim_encoding` / `attn_query` / `attn_key` 的**梯度**，Adam 一步之后就永久 NaN
> （表现为 `loss: nan`）。因此 `social_attention()` 开头先做
> `hidden_state = torch.nan_to_num(hidden_state)`。

## 怎么跑

本目录**不跟踪** `DATA_BLOCK/` 与 `OUTPUT_BLOCK/`（前者可由 `classroom/raw/` 重建，
后者是训练产物）。所以从仓库克隆后先重建数据：

```bash
cd week03/Week3-TrajNet++-modified

python prepare_classroom.py                      # 从 classroom/raw/ 重建 DATA_BLOCK/

# 基线（原版 pool）：请到 ../Week3-TrajNet++/ 跑，本目录的 social 分支与原版一致
python run_classroom.py --type attn_social  --output attn
python run_classroom.py --type front_social --output front
```

> 注意 `run_classroom.py` 的 `--output` 只影响 checkpoint 文件名，不参与训练；
> 本次实验三份代码都用 `--output baseline`（配合不同的 `--type`），
> 以保证除 `--type` 外配置完全一致。

评价与指标汇总（在 `week03/` 下执行）：

```bash
python collect_metrics.py results/lstm_social_baseline.pkl \
                           results/lstm_attn_social_baseline.pkl \
                           results/lstm_front_social_baseline.pkl --splits val test
```

## 依赖

- Python 3.9 / 3.10（**不要**用 3.12+，旧版 PyTorch 没有对应 wheel）。
- 装 `requirements-classroom.txt`，**不要**执行 `pip install -e .`（原 `setup.py` 会把 torch 钉到 1.10.0）。
- 实测环境：Windows x64、Python 3.9.23、torch 1.12.1+cpu。
