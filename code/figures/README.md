# figures/ — 图片输出目录

本目录只接收 `scripts/run_experiments.py` 写出的 19 张图（`fig01`–`fig19`：EDA、模型对比、ROC/PR、混淆矩阵、阈值与代价、重采样对照、特征重要性、学习曲线、PCA/t-SNE、任务 B/C 对比、效率权衡）。目录初始为空，运行后自动生成。

各步的中间图由该步自己的 `make_figures.py` 生成，**写在各自的 step 目录下**，不写入本目录：

| 命令 | 输出位置 |
|---|---|
| `python "unsupervised learning/step2/make_figures.py"` | `step2/fig_step2_feature_sets.png` |
| `python "unsupervised learning/step3/make_figures.py"` | `step3/fig_step3_control.png` |
| `python "unsupervised learning/step4/make_figures.py"` | `step4/fig_step4_ladder_and_modes.png` |
| `python "unsupervised learning/step5/make_figures.py"` | `step5/fig_step5_matrix_and_labels.png` |
| `python "unsupervised learning/step6/make_figures.py"` | `step6/fig_step6_cost_and_reason.png` |
| `python "unsupervised learning/step6after/make_figures.py"` | `step6after/fig_a1_multiseed.png` 等 3 张 |

图片只是结果的呈现，不参与任何后续计算；删除后重跑即可重建。最终报告中的 6 张插图已嵌入 `slides and report/Final_Project_Report.pdf`，不依赖本目录。
