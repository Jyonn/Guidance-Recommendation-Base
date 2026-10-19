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
