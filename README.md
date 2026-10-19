# MovieLens-100K 预处理

仅依赖 Python 3 标准库。运行：

```sh
python3 preprocess.py
```

原始数据位于 `data/raw/u.data`（本次从同级 Haonan/ml-100k 复制）。如需重新获取，请从 [GroupLens](https://grouplens.org/datasets/movielens/100k/) 下载并解压，将 `u.data` 放到此处。

处理规则：

- score < 3 为真实负样本（label=0），score > 3 为正样本（label=1），剔除 score=3，不生成额外负样本。
- 先剔除全部 score=3 交互，再按用户分组并按 timestamp 升序排列；时间相同时按 item_id 升序，再以原始行序兜底，与学生实现对齐。
- 对每个用户剔除 score=3 后的 N 条记录，训练集为 `[0, floor(0.8N))`，验证集长度为 `floor(0.1N)`（从训练集末尾接续），测试集为剩余记录。整数取整及后续冷启动物品过滤会使最终比例偏离 8:1:1。
- 完成划分后，以训练集中实际出现的 item 为白名单，删除验证/测试集中的冷启动物品交互；不重新划分，也不补充样本。
- 保留原始 user_id、item_id（从 1 开始）和 Unix 时间戳，不重编码。

输出为 `data/processed/train.csv`、`validation.csv`、`test.csv`，字段为 `user_id,item_id,score,timestamp,label`。`stats.json` 记录数据指纹和样本统计。

Git 维护代码、说明和统计；原始及派生数据仅保存在本地并被忽略，不随仓库分发。

## 基础模型训练

建议 Python 3.10+，安装依赖后直接运行：

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py --model mf --neg-mode real --loss bce --output-dir runs/mf-real-bce
python train.py --model ncf --neg-mode random --loss bpr --dim 32 --hidden 64 32 --lr 0.001 --batch-size 1024 --epochs 100 --patience 5 --output-dir runs/ncf-random-bpr
```

- 模型：`mf` 为用户/物品 embedding 内积；`ncf` 为 embedding 拼接后接 MLP，不含侧信息或额外网络。
- 负样本方式：`real` 使用真实正负交互；`random` 仅保留训练正交互，每个正样本抽取 `--negatives` 个负物品。候选为训练集物品全集减去该用户全部训练交互，每轮重新采样、有放回。不使用验证/测试信息，未观测物品可能含未来正交互。
- 损失：`bce` 使用 logits 的二元交叉熵；`mse` 使用 sigmoid 分数与 0/1 标签的均方误差；`bpr` 使用 `softplus(negative_score-positive_score)`。BPR 对每个正样本在同一用户的负物品池抽取 `--negatives` 个配对；没有正样本或没有候选负物品的用户不参与 BPR。随机模式无候选负物品时，点式损失仅保留正样本。
- 验证与测试：两种训练方式均使用真实标签。BCE/MSE 使用全部交互；BPR 使用各集合内部真实正负交互，固定种子生成配对，无可配对用户不参与；没有任何可用样本时明确报错。报告的是对应损失，不是 Recall/NDCG，三种损失的数值不可直接横向比较。
- Early stopping：每轮检查验证损失，下降超过 `--min-delta` 才算改进；连续 `--patience` 轮未改进则停止，恢复最佳权重后仅评估一次测试集。
- 可配置参数：`--lr`、`--weight-decay`、`--dim`、`--hidden`、`--dropout`、`--batch-size`、`--epochs`、`--patience`、`--min-delta`、`--negatives`、`--seed`、`--threads`、`--device`。默认 CPU；用 `python train.py --help` 查看默认值。
- `--data-dir` 默认为本项目 `data/processed`。训练时将原始 ID 映射为连续索引，验证/测试必须已完成冷物品过滤。
- 每次运行请指定独立 `--output-dir`，同目录会覆盖结果。`best.pt` 包含最佳权重、模型配置及原始用户/物品 ID 映射；`metrics.json` 包含配置、每轮损失、最佳轮次和测试损失。`runs/` 不纳入 Git。

运行基础验证（含 12 种配置和 early stopping）：

```sh
python -m unittest discover -s tests -v
```

已在 Python 3.11 / PyTorch 2.14.1 / CPU 环境验证：12 种组合均完成完整 MovieLens 数据上的单轮训练；单元测试覆盖采样排除规则、BPR 损失方向、全部组合及最佳权重恢复。默认 MF + real + BCE 实跑于第 36 轮 early stop，最佳轮次 31，验证 BCE 0.460639，测试 BCE 0.471653（seed=42）。

并行运行 10 组实验（MF/NCF × BCE/MSE/BPR × real/sampling，排除 BPR+real）：

```sh
.venv/bin/python run_experiments.py
```

每组一个独立 CPU 进程、一个计算线程，使用训练入口的默认超参数（seed=42、LR=0.001、dim=32、batch=1024、最多100轮、patience=5、min_delta=0.0001、每正样本1个负样本）。sampling 对应训练入口的 `--neg-mode random`。结果保存到新的 `runs/parallel-10-时间戳/`；每组包含日志、最佳模型和指标，根目录包含代码快照、数据指纹、运行命令及 `summary.csv/json`。
