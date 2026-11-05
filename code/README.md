# 代码与复现指南

本目录是最终报告中全部数值的生成源。数据已随包附带，**无需联网、无需额外文件**，拷到任意路径即可运行。

```bash
cd code
pip install -r requirements.txt
python scripts/run_experiments.py     # 先跑这条，任务 A/B/C 的主结果
```

---

## 1. 环境

| 项 | 值 |
|---|---|
| Python | 3.12.3（64-bit，Windows 实测；3.10+ 均可） |
| 主要依赖 | scikit-learn 1.6.1、pandas 2.2.3、numpy 2.2.4、scipy 1.15.2、matplotlib 3.10.1、seaborn 0.13.2 |
| 硬件 | 普通笔记本即可，全流程 CPU、无 GPU |
| 全流程耗时 | 3–8 分钟（随机器负载；本机两次实测 2 分 50 秒与约 8 分钟，主实验约 1 分 48 秒） |

所有随机过程使用固定种子（主实验与各步均为 `seed = 42`，无监督显著性检验另用 10 个固定种子 42 / 7 / 13 / 21 / 101 / 202 / 303 / 404 / 505 / 606），结果可逐位复现。

---

## 2. 目录结构

```
code/
├── data/
│   ├── ai4i2020.csv               原始数据（10,000 × 14，UCI AI4I 2020）
│   └── data_dictionary.md         字段说明与标签定义
├── scripts/
│   ├── eda.py                     数据探索（分布、缺失、相关性）
│   ├── run_experiments.py         任务 A/B/C 主实验：5 + 4 + 3 个模型、重采样对照、阈值与代价曲线
│   └── add_taskB_models.py        任务 B 补充两个模型（LR-OvR、KNN），写入独立 CSV 以免改动冻结结果
├── unsupervised learning/
│   ├── step1/features.py          特征层：F1_raw / F2_physics / F3_physics_pt 三套特征集（唯一入口）
│   ├── step2/run_step2.py         受控消融：同一检测器 + 同一阈值策略，只换特征集
│   ├── step3/run_step3.py         纯统计对照：不使用任何领域/物理知识的特征与流程
│   ├── step4/run_step4.py         支持边界扫描；support.py 为打分与阈值工具
│   ├── step5/run_step5.py         标签用量阶梯 L0 / L1 / L2；detectors.py 为 8 个检测器池
│   ├── step6/run_step6.py         决策封装：机理归因 + 代价敏感动作；pipeline.py 为端到端管线，demo.py 为单条样本演示
│   └── step6after/run_step6after.py  五项改进：多种子显著性、三档动作、融合、分组阈值、无标签估率
├── baseline/                      冻结基线参照（step4/step6 读取它做对照，勿删）
│   ├── baseline_metrics_all.csv
│   └── results/                   results_taskC_anomaly.csv、summary.json
├── results/                       本次提交对应的最终结果（见 §4）
└── figures/                       图片输出目录（脚本运行后写入）
```

每个 `stepN/README.md` 记录该步的方法、协议与结论，可作为复现时的阅读材料。

---

## 3. 运行顺序

脚本之间存在依赖（`step1/features.py` 是公共特征层，`step4` 读取 `step2` 的结果与 `baseline/` 的参照值，`step6` 读取 `baseline/results/summary.json`），因此请按下列顺序执行：

| 顺序 | 命令 | 作用 | 输出 | 耗时 |
|---|---|---|---|---|
| 1 | `python scripts/run_experiments.py` | 任务 A 二分类 5 模型、重采样对照、阈值与代价曲线、特征重要性、学习曲线、PCA/t-SNE；任务 B 多标签 4 模型；任务 C 无监督 3 检测器 | `results/*.csv`、`results/summary.json`、`figures/fig01–fig19` | ≈1 分 48 秒 |
| 2 | `python scripts/add_taskB_models.py` | 任务 B 补足到 6 个模型（课程要求每任务 ≥5） | `results/results_taskB_extra_models.csv` | <5 秒 |
| 3 | `python "unsupervised learning/step2/run_step2.py"` | 特征集受控消融（F1_raw / F2_physics / F3_physics_pt × 3 检测器） | `step2/results_step2.csv` | <1 分钟 |
| 4 | `python "unsupervised learning/step3/run_step3.py"` | 纯统计对照组（多项式展开、Nystroem 等，不含物理量） | `step3/results_step3.csv` | <1 分钟 |
| 5 | `python "unsupervised learning/step4/run_step4.py"` | 支持边界扫描与参考点对齐 | `step4/results_step4_*.csv` | <1 分钟 |
| 6 | `python "unsupervised learning/step5/run_step5.py"` | 标签用量阶梯 L0 / L1 / L2 全矩阵 | `step5/results_step5.csv` | ≈1 分 31 秒 |
| 7 | `python "unsupervised learning/step6/run_step6.py"` | 机理归因、代价敏感动作与端到端管线 | `step6/results_step6*.csv` | ≈35 秒 |
| 8 | `python "unsupervised learning/step6after/run_step6after.py"` | 10 划分显著性、三档动作盈亏平衡、子空间融合、分组阈值、无标签估率 | `step6after/results_a1…a5*.csv`、`summary_step6after.json` | ≈47 秒 |

> 脚本内部用 `Path(__file__).resolve().parent…` 推导项目根目录，**请勿单独移动某个脚本**，保持当前层级即可。中文目录名 `unsupervised learning` 在命令行中需加引号。

---

## 4. 预期结果（用于核对复现是否成功）

### 4.1 任务 A 二分类（测试集 3,000 条，含 102 条故障）

| 模型 | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Decision Tree | 0.9420 | 0.3333 | 0.7059 | 0.4528 | 0.8368 | 0.4933 |
| Random Forest | 0.9773 | 0.8542 | 0.4020 | 0.5467 | 0.9658 | 0.7181 |
| SVM (RBF) | 0.9367 | 0.3295 | 0.8333 | 0.4722 | 0.9697 | 0.6492 |
| **MLP** | **0.9813** | 0.7875 | 0.6176 | **0.6923** | **0.9797** | **0.7471** |
| Logistic Regression | 0.8313 | 0.1431 | 0.7941 | 0.2425 | 0.8885 | 0.3799 |

### 4.2 任务 B 多标签（6 模型，指标为 Micro/Macro）

| 模型 | Micro-F1 | Macro-F1 | Exact Match | Hamming Loss |
|---|---|---|---|---|
| **MLP** | **0.7464** | **0.5168** | 0.9823 | 0.0035 |
| Decision Tree | 0.6923 | 0.5045 | 0.9797 | 0.0043 |
| LR (OvR) | 0.6667 | 0.4751 | 0.9797 | 0.0041 |
| Random Forest | 0.5868 | 0.4032 | 0.9780 | 0.0046 |
| SVM (RBF) | 0.4269 | 0.4669 | 0.9257 | 0.0159 |
| KNN (k=5) | 0.2979 | 0.2066 | 0.9693 | 0.0066 |

### 4.3 任务 C 无监督（基线，contamination 先验 3.39%，阈值不碰测试标签）

| 检测器 | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|
| Isolation Forest | 0.1244 | 0.7570 | 0.1049 |
| One-Class SVM | 0.2029 | 0.7637 | 0.1675 |
| Local Outlier Factor | 0.1800 | 0.8132 | 0.1497 |

> 报告 **Table 4**（八个检测器同操作点比较）对应 `step5/results_step5.csv` 中 `layer = 'Lref ctm=3.39%'` 且 `feature_set = 'F2_physics'` 的那 8 行（MSV 0.3832、Mahalanobis 0.2959、HBOS 0.2870、KDE 0.2636、LOF 0.2545、iForest 0.2439、GMM 0.2167、OCSVM 0.2150）。
> 报告 **Table 5**（40 个被报警真故障的原因码归属）对应 `step6/results_step6_contrib.csv`。

### 4.4 关键阶梯（报告的核心论证链）

| 环节 | 脚本 | 零标签 F1 | 说明 |
|---|---|---|---|
| 原始特征 + 稀有度打分 | step2 (`F1_raw`) | 0.2029–0.2095 | 与冻结基线一致 |
| 只换特征（加 3 个物理量） | step2 (`F2_physics`) | 0.2439 | 特征贡献 +0.04 |
| 支持边界扫描 | step4 | 0.2439 | 阈值策略尚不足以解释全部差距 |
| 换打分口径（L0，纯无标签） | step5 (`Lref ctm=3.39%`) | 0.3832 | 打分口径贡献（9 子空间池） |
| **只保留 3 个约束单坐标** | step6after (`a1`, `MSV_singletons`) | **0.4322** | 报告 Figure 4 的第三根柱 |
| L2 用 300 条随机标签校准阈值 | step5 (`L2b rate from M=300`) | 0.4348 | 标签值多少钱的量化答案 |
| 剔除 HDF 子空间 + 多划分均值 | step6after (`a1`) | **0.4631 ± 0.0321** | 10 划分均值；PR-AUC 0.5164 |
| 有监督上界 | 任务 A (MLP) | 0.6923 | 标签用于学机理才拿得到 |

> step5/step6after 的结果表除指标列外还含 `fit_time_s` / `train_time_s` 等耗时列，**该列随机器负载浮动是正常的**，其余数值应逐位一致。

---

## 5. 协议要点（复现时需要知道的口径）

1. **划分**：分层 70/30，`random_state=42`，测试集 3,000 条（含 102 条故障）；所有步骤共用同一划分。
2. **评估真值**：统一使用 `Machine failure`；同时报告 ROC-AUC 与 PR-AUC。
3. **标签隔离**：标签只用于评估。任何一步的拟合、选型、定阈值都不使用测试标签；唯一例外是显式标注为 `oracle` 的行（仅作性能上界参照）。
4. **预处理**：标准化等变换只在训练集上拟合，再作用于测试集。
5. **阈值口径**：无监督方法默认按 contamination 先验 3.39% 定阈值（等价于固定报警预算），L1 用约 1% 污染率先验，L2 用 10/20/50 个已知故障或 100/300/1000 条随机标签校准阈值。
6. **代价模型**：漏检 1000、误报 100（10:1 非对称），三档动作另设监控成本 C_MON 并给出盈亏平衡条件 α ≥ 0.68。
7. **编码**：数据为 UTF-8 with BOM，读取时用 `encoding='utf-8-sig'`；绘图字体优先 Microsoft YaHei / SimHei，缺失时回退 DejaVu Sans（中文图中文字形会受影响，数值不受影响）。

---

## 6. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `FileNotFoundError: data/ai4i2020.csv` | 脚本被移动到了别的位置；保持 `scripts/` 与 `data/` 同级 |
| 数值与本文表格在小数第 4 位不同 | 检查 scikit-learn 版本；1.6.x 下与本文完全一致，版本差异可能带来随机初始化差异（MLP 尤为敏感） |
| `TrainTime(s)` 与本文不同 | 正常，属于机器负载差异，非结果差异 |
| 中文图显示方框 | 缺少 Microsoft YaHei / SimHei 字体，安装字体或忽略（不影响数值） |
| 想核对报告里的某个数 | 见仓库根 `README.md` 的「报告 ↔ 结果文件」表 |
