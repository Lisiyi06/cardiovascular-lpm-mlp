# 基于多层感知机（MLP）的心血管集中参数模型（Lumped Parameter Model, LPM）参数逆向推理工具。

任务目标：输入临床可测量的心脏生理指标（心室容积、射血分数、平均动脉压等），通过训练好的 MLP 替代模型，快速预测个性化的 LPM 参数，从而在 MATLAB/Simulink 中生成患者特定的左心室压力-容积（PV）环。

---

## 项目架构

```
.
├── LPM_model.slx               # Simulink 集中参数模型（核心仿真）*成果尚未发布暂时无法公开
├── LHS.m                        # MATLAB 脚本：拉丁超立方采样 + 批量仿真生成数据集
├── synthetic_dataset.mat        # 生成的仿真数据集（969 组有效样本）
│
├── model_train.py               # MLP 替代模型训练（PyTorch）
├── usage.py                     # 用户交互推理脚本：输入生理指标 → 输出 LPM 参数
│
├── param/                       # 参考参数文件
│   ├── simulink dafult params.txt   # Simulink 模型默认结构体参数
│   └── usage_params.txt             # usage.py 预置的用户工况参数
│
├── MLP_result/                  # 完整训练产物（被 usage.py 直接调用）
│   ├── best_lpm_model.pt        #   最佳模型权重 + 归一化器
│   ├── loss_curve.png           #   训练/验证损失曲线
│   └── test_predictions.npz     #   验证集预测结果
│
├── test_result/                 # 小规模调试测试产物
│   ├── best_lpm_model.pt
│   ├── loss_curve.png
│   └── test_predictions.npz
│
├── 多层感知机算法架构说明.md      # 算法架构详细文档
├── environment.yaml               # conda 环境导出文件
├── 心室的压力容积环.docx          # param中给出的5组样例工况生成PV环结果与分析
├── 项目演示案例.mp4               # 运行本项目的演示视频 *成果尚未发布暂时无法公开
└── README.md
```

---

## 背景

集中参数模型（LPM）是心血管系统仿真的经典方法，其参数（如心室弹性、血管阻力、顺应性等）直接决定了仿真输出的 PV 环形态。传统方法需要手动调参，费时且依赖经验。

本项目提出一种**数据驱动的逆映射方法**：

1. 使用 LHS（拉丁超立方采样）在 LPM 参数空间均匀采样；
2. 批量运行 Simulink 仿真获得对应的生理指标输出；
3. 训练 MLP 回归模型学习从 **生理指标 → LPM 参数** 的映射；
4. 用户只需输入临床检测数据，模型即可输出个性化LPM参数，直接用于 Simulink 仿真，为患者提供快速且个性化的仿真结果。

---

## 数据说明

### 数据集 (`synthetic_dataset.mat`)

由 `LHS.m` 脚本生成，包含两个变量：

| 变量      | 形状     | 说明                      |
| --------- | -------- | ------------------------- |
| `X_clean` | `(N, 6)` | 6 维生理指标（输入特征）  |
| `Y_clean` | `(N, 5)` | 5 维 LPM 参数（预测目标） |

**输入特征（X 的 6 列）**：

| 列  | 名称  | 含义           | 单位       |
| --- | ----- | -------------- | ---------- |
| 0   | LVEDV | 左室舒张末容积 | mL         |
| 1   | LVESV | 左室收缩末容积 | mL         |
| 2   | LVEF  | 左室射血分数   | 小数 (0~1) |
| 3   | RVEDV | 右室舒张末容积 | mL         |
| 4   | RVESV | 右室收缩末容积 | mL         |
| 5   | MAP   | 平均动脉压     | mmHg       |

**预测目标（Y 的 5 列）**：

| 列  | 名称   | 含义             | 单位      | 采样范围     |
| --- | ------ | ---------------- | --------- | ------------ |
| 0   | Elvmax | 左室最大弹性     | mmHg/mL   | [1.5, 4.0]   |
| 1   | Ervmax | 右室最大弹性     | mmHg/mL   | [0.6, 2.0]   |
| 2   | R_sar  | 体循环小动脉阻力 | mmHg·s/mL | [0.5, 2.5]   |
| 3   | C_ao   | 主动脉顺应性     | mL/mmHg   | [0.04, 0.15] |
| 4   | TBV    | 总血容量         | mL        | [700, 1200]  |

---

## 依赖环境

### Python（模型训练 & 推理）

本项目统一使用 `sci_computing` conda 环境，已导出为 `environment.yaml`。

一键复现：

```bash
conda env create -f environment.yaml
conda activate sci_computing
```

### MATLAB（数据生成 & 仿真）

- MATLAB（带 Simulink），推荐版本：MATLAB R2024-a

---

## 运行方式

整个流程分为 **4 步**，按顺序执行。第 2 步可跳过（数据集已生成）。

### 第 1 步：Simulink 模型单次测试（可选）

在 MATLAB 中打开 `LPM_model.slx`，手动设置 `base_params` 结构体后运行，查看 PV 环输出。

- 默认参数文件见 `param/simulink dafult params.txt`
- 可直接复制粘贴到 MATLAB 命令行

### 第 2 步：批量仿真生成数据集（可选）

在 MATLAB 中运行 `LHS.m`：

```matlab
>> LHS
```

脚本将：
- 对 5 个可变参数进行拉丁超立方采样（1000 组）
- 自动运行 Simulink 批量仿真
- 提取稳态周期中的生理指标
- 过滤非生理结果（如 LVEDV ≤ 0、LVEF < 0.1 等）
- 保存 `synthetic_dataset.mat`

> 默认已包含生成好的数据集（969 组有效样本），可直接用于训练。

### 第 3 步：训练 MLP 替代模型

```bash
conda activate sci_computing

# 完整模式（使用 synthetic_dataset.mat）
python model_train.py

# 测试模式（使用 5 组内置数据，用于调试）
python model_train.py --mode test

# 常用参数覆盖
python model_train.py --mode full --epochs 800 --lr 5e-4 --batch-size 64
```

训练完成后输出到当前目录：
- `best_lpm_model.pt` — 模型权重 + 归一化器
- `loss_curve.png` — 损失曲线图
- `test_predictions.npz` — 验证集预测

建议将产物移入 `MLP_result/` 目录供 `usage.py` 调用（默认路径即 `MLP_result/best_lpm_model.pt`）。

### 第 4 步：用户交互推理

```bash
conda activate sci_computing
python usage.py
```

程序将提示输入 6 项生理指标：

```
LVEDV  - 左室舒张末容积 (mL)
LVESV  - 左室收缩末容积 (mL)
LVEF   - 左室射血分数
RVEDV  - 右室舒张末容积 (mL)
RVESV  - 右室收缩末容积 (mL)
MAP    - 平均动脉压 (mmHg)
```

此外还支持：
- **心率输入**：自动换算为心动周期 T
- **身高体重估算 TBV**：使用 Nadler 公式替代 MLP 的预测值

程序输出 MATLAB `struct` 代码，复制粘贴到 MATLAB 命令行即可运行 Simulink 仿真。

### 预置工况

`param/usage_params.txt` 中准备了 **5 组典型工况**：

| 工况 | 描述       |
| ---- | ---------- |
| 1    | 正常成年人 |
| 2    | 高血压     |
| 3    | 心力衰竭   |
| 4    | 主动脉狭窄 |
| 5    | 肺动脉高压 |

---

## MLP 网络架构

```
输入层 (6)
    ↓ Linear(6→128) + BatchNorm1d + ReLU + Dropout(0.2)
隐藏层 1 (128)
    ↓ Linear(128→64) + BatchNorm1d + ReLU + Dropout(0.2)
隐藏层 2 (64)
    ↓ Linear(64→32) + BatchNorm1d + ReLU + Dropout(0.2)
隐藏层 3 (32)
    ↓ Linear(32→5)  无激活
输出层 (5)
```

- **漏斗形设计**：逐层降维，提取高层次抽象特征
- **BatchNorm1d**：加速收敛、稳定训练
- **Dropout(0.2)**：防止过拟合
- **输出层无激活**：允许预测任意连续实数值

训练配置：
- 优化器：Adam（lr=1e-3, weight_decay=1e-5）
- 损失函数：MSE Loss
- 早停策略：验证损失连续 50 个 epoch 未改善则停止
- 最大 epoch：500
- 训练/验证划分：80% / 20%

---

## 输出文件说明

| 路径                               | 内容                          | 生成方式                     |
| ---------------------------------- | ----------------------------- | ---------------------------- |
| `MLP_result/best_lpm_model.pt`     | 最佳模型权重 + StandardScaler | `model_train.py`             |
| `MLP_result/loss_curve.png`        | 训练/验证损失曲线             | `model_train.py`             |
| `MLP_result/test_predictions.npz`  | 验证集真实值 vs 预测值        | `model_train.py`             |
| `test_result/`                     | 同上，但使用 5 组内置测试数据 | `model_train.py --mode test` |
| `synthetic_dataset.mat`            | 969 组 LHS 仿真样本           | `LHS.m`                      |
| `param/simulink dafult params.txt` | Simulink 默认参数             | 手动维护                     |
| `param/usage_params.txt`           | 5 组预置用户工况              | 手动维护                     |

---

## 引用

本项目尚有疏漏，仅用于作者团队深度科研训练srtp项目；欢迎联系作者。

