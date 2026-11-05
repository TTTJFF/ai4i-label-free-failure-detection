# Step 1 — 特征层（Feature Layer）

**状态**：完成。无标签、确定性、可复现。
**产出**：`features.py`（唯一入口）

---

## 1. 这一层做什么

把原始的 5 个传感器读数变换成三套特征集，供 step 2 做受控消融。

| 特征集 | 组成 | 维度 | 说明 |
|---|---|---|---|
| `F1_raw` | 5 个原始传感器 + 机型（drop-first 独热） | 7 | **与冻结基线完全一致的表示**，作为对照 |
| `F2_physics` | F1 + 3 个派生物理量 | 10 | 本次假设检验的核心 |
| `F3_physics_pt` | F2 + 5 个「按机型标准化」的传感器 | 15 | 检验机型阈值差异带来的增益 |

原始 5 个传感器：`Air temperature [K]`、`Process temperature [K]`、`Rotational speed [rpm]`、`Torque [Nm]`、`Tool wear [min]`。

## 2. 三个派生物理量（本层的核心）

这三个量不是随手加的，而是**失效机理实际定义在其上的物理量**。2026-10-11 的只读规则验证表明，五个失效模式中有三个是这些量的确定性函数（全量 10,000 条零误报）：

| 派生特征 | 计算式 | 物理含义 | 对应失效模式 |
|---|---|---|---|
| `Power [W]` | `Torque × 2π × Speed / 60` | 机械功率 | PWF 功率故障（功率落在 [3500, 9000] W 之外） |
| `Temp diff [K]` | `Process temp − Air temp` | 散热裕度 | HDF 散热故障（温差 < 8.6 K 且 转速 < 1380 rpm） |
| `Overstrain [Nm*min]` | `Torque × Tool wear` | 累计机械应变 | OSF 过载故障（超过阈值 L 11000 / M 12000 / H 13000） |

**为什么这一步是关键**：在原始坐标系里，这三条约束都是**乘积形式的半空间**（`T × ω`、`T × W` 是两个坐标相乘），而 iForest / LOF / One-Class SVM 这类方法只能做轴对齐切分，学不出乘积。把乘积提升为显式坐标之后，约束就变成了区间约束。

全量数据上的取值范围（供核对）：

| 派生特征 | min | 25% | 50% | 75% | max |
|---|---|---|---|---|---|
| `Power [W]` | 1148.44 | 5561.18 | 6271.03 | 7003.00 | 10469.92 |
| `Temp diff [K]` | 7.60 | 9.30 | 9.80 | 11.00 | 12.10 |
| `Overstrain [Nm*min]` | 0.00 | 1963.65 | 4012.95 | 6279.00 | 16497.00 |

## 3. 无标签声明（对应决议 Q1）

本文件**不读取** `Machine failure`、也不读取五个模式标志来做任何变换。所有派生量都是物理量的确定性计算，按机型标准化只用训练集的均值/方差。

这属于决议 Q1 允许的范围：领域知识进入**特征构造**，属于引入合理的归纳偏置；学习环节本身仍在无标签数据上探索结构。

## 4. 两个工程细节

1. **按机型标准化的统计量只在训练集上估计**（`fit_per_type_stats(df_train)`），再应用到测试集。若用全量数据估计，测试集的信息会渗进特征。
2. **机型编码用 drop-first**（保留 `Type=M`、`Type=H`，L 为基准），与冻结基线的 `OneHotEncoder(drop='first')` 一致，这样 `F1_raw` 才能和基线直接对比。

## 5. 用法

```python
import sys; sys.path.insert(0, 'unsupervised learning/step1')
import features as F

df = F.load_data()
frames, y = F.build_frames(df, train_mask)   # train_mask 用于估计按机型标准化统计量
print(frames['F2_physics'].shape)
```

自检：

```bash
python "unsupervised learning/step1/features.py"
# rows 10000, failures 339 (3.39%)
# F1_raw shape=(10000, 7)   F2_physics shape=(10000, 10)   F3_physics_pt shape=(10000, 15)
```
