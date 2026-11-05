# 数据字典 — AI4I 2020 Predictive Maintenance Dataset

- 文件：`ai4i2020.csv`（522,048 字节）
- 来源：UCI Machine Learning Repository, ID 601
- 引用：Matzka, S. (2020). AI4I 2020 Predictive Maintenance Dataset [Data set].
  UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/601
- DOI：10.24432/C5HS5C
- 许可：CC BY 4.0

## 字段

| 字段 | 类型 | 说明 | 本项目中是否使用 |
|---|---|---|---|
| UDI | 整数 | 行号 1–10000 | 否（丢弃） |
| Product ID | 字符串 | 产品唯一标识（L/M/H + 序号） | 否（丢弃，避免泄漏） |
| Type | 类别 | 机型 L / M / H | 是（独热编码） |
| Air temperature [K] | 浮点 | 环境温度 | 是 |
| Process temperature [K] | 浮点 | 过程温度 | 是 |
| Rotational speed [rpm] | 整数 | 转速 | 是 |
| Torque [Nm] | 浮点 | 扭矩 | 是 |
| Tool wear [min] | 整数 | 累计刀具磨损 | 是 |
| Machine failure | 0/1 | 总体故障标签，**任务 A 目标** | 是 |
| TWF | 0/1 | 刀具磨损故障 | 是（任务 B） |
| HDF | 0/1 | 散热故障 | 是（任务 B） |
| PWF | 0/1 | 功率故障 | 是（任务 B） |
| OSF | 0/1 | 过载故障 | 是（任务 B） |
| RNF | 0/1 | 随机故障 | 是（任务 B） |

## 关键统计（实测）

- 样本 10,000 条，14 列，无缺失值
- 总体故障 339 条（3.39%）
- 五模式计数：TWF 46、HDF 115、PWF 95、OSF 98、RNF 19，合计 373（> 339，存在多标签重叠）
- 机型分布：L 6,000、M 2,997、H 1,003
- 划分：分层 70/30，seed 42 → 训练 7,000 条（故障 237）/ 测试 3,000 条（故障 102）
- **数据集为合成数据**，用于模拟真实产线失效机理；结论外推到真实设备时需注明该限制
