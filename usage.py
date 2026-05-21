import torch
import torch.nn as nn
import numpy as np
import os

# ==================== 模型定义（与训练时完全一致） ====================
class MLP(nn.Module):
    def __init__(self, input_dim=6, output_dim=5, hidden_dims=[128, 64, 32], dropout=0.2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ==================== 默认基础参数 ====================
BASE_PARAMS_DEFAULT = {
    'R_ao': 0.003,
    'L_ao': 0.0000168,
    'R_sat': 0.05,
    'L_sat': 0.0017,
    'C_sat': 1,
    'R_svn': 0.075,
    'C_svn': 20.5,
    'T': 0.8,              # 心动周期 (s)，默认 75 bpm
    'R_pas': 0.002,
    'C_pas': 0.18,
    'L_pas': 0.000052,
    'R_pat': 0.01 + 0.04,
    'C_pat': 3.8,
    'L_pat': 0.00017,
    'R_pcp': 0.05,
    'R_pvn': 0.006,
    'C_pvn': 20.5,
}

# 可变参数名称（顺序与训练数据 Y_clean 列一致）
VARIABLE_PARAMS = ['Elvmax', 'Ervmax', 'R_sar', 'C_ao', 'TBV']

# 模型输入特征名称（顺序与训练数据 X_clean 列一致）
FEATURE_NAMES = ['LVEDV (mL)', 'LVESV (mL)', 'LVEF', 'RVEDV (mL)', 'RVESV (mL)', 'MAP (mmHg)']

# 加载模型文件路径
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'MLP_result', 'best_lpm_model.pt') #项目文件结构，usage.py与MLP_reault文件夹在同一路径下


# ==================== 辅助函数 ====================
def load_model(pt_path=MODEL_PATH):
    """加载训练好的模型和归一化器。"""
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"模型文件 {pt_path} 不存在，请先训练模型。")
    checkpoint = torch.load(pt_path, map_location='cpu', weights_only=False)
    model = MLP()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    scaler_X = checkpoint['scaler_X']
    scaler_Y = checkpoint['scaler_Y']
    return model, scaler_X, scaler_Y


def safe_float_input(prompt, default=None):
    """带默认值的浮点数输入，直接回车使用默认值。"""
    if default is not None:
        user_input = input(f"{prompt} [?=默认 {default}]: ").strip()
    else:
        user_input = input(f"{prompt}: ").strip()

    if user_input == '' or user_input == '?':
        if default is not None:
            print(f"  → 使用默认值: {default}")
            return default
        else:
            print("  ! 此项为必填，请输入数值。")
            return safe_float_input(prompt, default)
    try:
        return float(user_input)
    except ValueError:
        print("  ! 输入无效，请输入一个数字。")
        return safe_float_input(prompt, default)


def estimate_tbv(height_cm=None, weight_kg=None, gender=None):
    """
    根据身高体重估算总血容量 (TBV)。
    使用 Nadler 公式（成人）:
      - 男性: TBV (mL) = 0.3669 * H³ + 0.03219 * W + 0.6041
      - 女性: TBV (mL) = 0.3561 * H³ + 0.03308 * W + 0.1833
      其中 H = 身高(m), W = 体重(kg)
    """
    if height_cm is None or weight_kg is None:
        return None

    h_m = height_cm / 100.0
    if gender is None or gender.lower() not in ['m', 'f', 'male', 'female']:
        # 默认使用男性公式
        tbv_l = 0.3669 * (h_m ** 3) + 0.03219 * weight_kg + 0.6041
    elif gender.lower() in ['m', 'male']:
        tbv_l = 0.3669 * (h_m ** 3) + 0.03219 * weight_kg + 0.6041
    else:
        tbv_l = 0.3561 * (h_m ** 3) + 0.03308 * weight_kg + 0.1833

    return tbv_l * 1000  # 转换为 mL


def format_matlab_struct(params_dict):
    """将参数字典格式化为 MATLAB struct 代码。"""
    lines = ["base_params = struct(..."]
    keys = list(params_dict.keys())
    for i, key in enumerate(keys):
        val = params_dict[key]
        # 格式化数值
        if isinstance(val, float):
            val_str = f"{val:.6g}"
        else:
            val_str = str(val)
        # 判断是否为最后一个参数
        if i == len(keys) - 1:
            lines.append(f"    '{key}', {val_str});")
        else:
            lines.append(f"    '{key}', {val_str}, ...")
    return '\n'.join(lines)


# ==================== 主流程 ====================
def main():
    print("=" * 60)
    print("  LPM 参数推理系统 —— 从生理指标到 LPM 参数")
    print("  基于 MLP 替代模型")
    print("=" * 60)
    print()

    # ---- 1. 加载模型 ----
    print("[1/3] 加载训练好的模型...")
    try:
        model, scaler_X, scaler_Y = load_model(MODEL_PATH)
        print(f"  ✓ 模型 {MODEL_PATH} 加载成功\n")
    except Exception as e:
        print(f"  ✗ 加载失败: {e}")
        return

    # ---- 2. 收集用户输入 ----
    print("[2/3] 请输入患者的生理指标\n")
    print("  ── 必要输入（模型核心特征）──")
    
    lvedv = safe_float_input("  LVEDV  - 左室舒张末容积 (mL)")
    lvesv = safe_float_input("  LVESV  - 左室收缩末容积 (mL)")
    lvef  = safe_float_input("  LVEF   - 左室射血分数 (小数，如 0.55)")
    rvedv = safe_float_input("  RVEDV  - 右室舒张末容积 (mL)")
    rvesv = safe_float_input("  RVESV  - 右室收缩末容积 (mL)")
    map_val = safe_float_input("  MAP    - 平均动脉压 (mmHg)")

    print("\n  ── 可选输入（直接回车使用默认值）──")

    # 心率 → 心动周期
    print("\n  心率 (HR) 用于计算心动周期 T。")
    print("    公式: T = 60 / HR")
    print("    默认: HR = 75 bpm, T = 0.8 s")
    hr = safe_float_input("  心率 HR (bpm)", default=75)
    T = 60.0 / hr

    # 身高体重 → TBV 估算
    print("\n  身高体重用于估算总血容量 (TBV)。")
    print("    公式: Nadler 公式（成人）")
    print("    默认: TBV = 950 mL（不使用公式估算时）")
    use_tbv_formula = input("  是否使用身高体重估算 TBV？(y/n, 默认 n): ").strip().lower()

    tbv_estimated = None
    if use_tbv_formula in ['y', 'yes']:
        height = safe_float_input("  身高 (cm)")
        weight = safe_float_input("  体重 (kg)")
        gender = input("  性别 (m/f, 默认 m): ").strip().lower()
        if gender == '':
            gender = 'm'
        tbv_estimated = estimate_tbv(height, weight, gender)
        if tbv_estimated is not None:
            print(f"  → 估算 TBV = {tbv_estimated:.1f} mL")
        else:
            print("  ! 估算失败，将使用模型预测值。")

    # ---- 3. 模型预测 ----
    print("\n[3/3] 正在预测 LPM 参数...")

    # 组装输入特征
    X_new = np.array([[lvedv, lvesv, lvef, rvedv, rvesv, map_val]], dtype=np.float32)

    # 归一化 → 预测 → 反归一化
    X_scaled = scaler_X.transform(X_new)
    with torch.no_grad():
        Y_scaled = model(torch.from_numpy(X_scaled)).numpy()
    Y_pred = scaler_Y.inverse_transform(Y_scaled)[0]

    # 构建预测量字典
    predicted_params = dict(zip(VARIABLE_PARAMS, Y_pred))
    print("  ✓ 预测完成\n")

    # ---- 4. 合并参数 ----
    # 以默认参数为底
    final_params = BASE_PARAMS_DEFAULT.copy()

    # 覆盖心动周期
    final_params['T'] = T

    # 覆盖 MLP 预测的 5 个参数（Elvmax, Ervmax, R_sar, C_ao, TBV）
    for key in VARIABLE_PARAMS:
        final_params[key] = predicted_params[key]

    # 如果用户用了 TBV 公式，用估算值覆盖 MLP 预测值
    if use_tbv_formula in ['y', 'yes'] and tbv_estimated is not None:
        print(f"  ℹ TBV 使用公式估算值 {tbv_estimated:.1f} 替代 MLP 预测值 {predicted_params['TBV']:.1f}")
        final_params['TBV'] = tbv_estimated

    # ---- 5. 输出结果 ----
    print("\n" + "=" * 60)
    print("  MATLAB 可用的 LPM 参数（可直接复制粘贴）")
    print("=" * 60)
    print()

    matlab_code = format_matlab_struct(final_params)
    print(matlab_code)

    print()
    print("=" * 60)
    print("  使用说明:")
    print("  1. 复制上方 base_params = struct(...) 代码")
    print("  2. 粘贴到 MATLAB 命令行")
    print("  3. 运行 sim('LPM_model', ...) 即可仿真")
    print("=" * 60)

    # 同时打印预测参数摘要
    print("\n  参数摘要:")
    print(f"    心动周期 T: {T:.4f} s (心率 {hr:.0f} bpm)")
    for key in VARIABLE_PARAMS:
        print(f"    {key}: {final_params[key]:.6g}")
    print()


if __name__ == '__main__':
    main()