import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无头后端，避免 GUI 依赖
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ============================================================================
# 配置字典（可在此处直接修改，也可通过命令行参数覆盖）
# ============================================================================
CFG = {
    "mode": "full",              # "test" | "full"
    "mat_path": "synthetic_dataset.mat",
    "seed": 42,
    "test_size": 0.2,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "max_epochs": 500,
    "patience": 50,              # 早停耐心值
    "hidden_dims": [128, 64, 32],
    "dropout": 0.2,
    "device": "auto",            # "auto" | "cpu" | "cuda"
    "save_dir": ".",
}

# 参数名称（用作输出标注）
PARAM_NAMES = ["Elvmax", "Ervmax", "R_sar", "C_ao", "TBV"]

# ---------------------------------------------------------------------------
# 内置测试数据（5 组，用于快速验证训练流程）
# ---------------------------------------------------------------------------
TEST_X = np.array(
    [
        [120.0, 50.0, 0.58, 130.0, 55.0, 90.0],
        [100.0, 40.0, 0.60, 110.0, 45.0, 85.0],
        [140.0, 70.0, 0.50, 150.0, 75.0, 95.0],
        [90.0, 30.0, 0.67, 100.0, 35.0, 80.0],
        [130.0, 60.0, 0.54, 140.0, 65.0, 92.0],
    ],
    dtype=np.float32,
)

TEST_Y = np.array(
    [
        [2.5, 1.2, 1.5, 0.08, 950.0],
        [2.8, 1.0, 1.2, 0.10, 900.0],
        [2.0, 1.5, 2.0, 0.06, 1000.0],
        [3.2, 0.8, 0.8, 0.12, 850.0],
        [2.3, 1.3, 1.7, 0.09, 970.0],
    ],
    dtype=np.float32,
)


# ============================================================================
# 1. 数据加载
# ============================================================================
def load_data(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """加载 X 和 Y 数据，返回 (X, Y) 均为 float32 的 numpy 数组。

    Raises:
        FileNotFoundError: 完整模式下 .mat 文件不存在。
    """
    if cfg["mode"] == "test":
        print("[数据] 测试模式：使用 5 组内置数据")
        return TEST_X.copy(), TEST_Y.copy()

    mat_path = Path(cfg["mat_path"])
    if not mat_path.is_file():
        raise FileNotFoundError(
            f"[错误] 数据文件 '{mat_path}' 不存在。\n"
            f"  请确认文件位置，或将 CFG['mode'] 设为 'test' 使用内置测试数据。"
        )

    print(f"[数据] 加载 '{mat_path}' ...")
    data = loadmat(str(mat_path))
    X = data["X_clean"].astype(np.float32)
    Y = data["Y_clean"].astype(np.float32)
    print(f"[数据] X: {X.shape}, Y: {Y.shape}")
    return X, Y


# ============================================================================
# 2. 模型定义
# ============================================================================
class MLPRegressor(nn.Module):
    """"宽-窄-窄" 漏斗形 MLP，带 BatchNorm + Dropout。"""

    def __init__(
        self,
        in_dim: int = 6,
        out_dim: int = 5,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev = h
        # 输出层：无激活
        layers.append(nn.Linear(prev, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================================
# 3. 数据预处理管线
# ============================================================================
def prepare_dataloaders(
    X: np.ndarray,
    Y: np.ndarray,
    cfg: dict,
) -> tuple[DataLoader, DataLoader, StandardScaler, StandardScaler]:
    """标准化、划分数据集、封装 DataLoader。返回 (train_loader, val_loader, scaler_X, scaler_Y)。"""
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X).astype(np.float32)
    Y_scaled = scaler_Y.fit_transform(Y).astype(np.float32)

    X_train, X_val, Y_train, Y_val = train_test_split(
        X_scaled, Y_scaled,
        test_size=cfg["test_size"],
        random_state=cfg["seed"],
    )

    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(Y_train),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(Y_val),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
    )

    print(f"[数据] 训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")
    return train_loader, val_loader, scaler_X, scaler_Y


# ============================================================================
# 4. 训练循环
# ============================================================================
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
    device: torch.device,
) -> tuple[nn.Module, list[float], list[float]]:
    """训练模型，返回 (best_model, train_losses, val_losses)。"""
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    criterion = nn.MSELoss()

    best_loss = float("inf")
    best_state = model.state_dict()
    epochs_no_improve = 0
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(1, cfg["max_epochs"] + 1):
        # ---- 训练 ----
        model.train()
        epoch_train_loss = 0.0
        for Xb, Yb in train_loader:
            Xb, Yb = Xb.to(device), Yb.to(device)
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, Yb)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * Xb.size(0)

        avg_train_loss = epoch_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # ---- 验证 ----
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for Xb, Yb in val_loader:
                Xb, Yb = Xb.to(device), Yb.to(device)
                pred = model(Xb)
                loss = criterion(pred, Yb)
                epoch_val_loss += loss.item() * Xb.size(0)

        avg_val_loss = epoch_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)

        # ---- 早停 ----
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # ---- 日志 ----
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Train Loss: {avg_train_loss:.6e} | Val Loss: {avg_val_loss:.6e}")

        if epochs_no_improve >= cfg["patience"]:
            print(f"[早停] 验证损失连续 {cfg['patience']} 个 epoch 未改善，停止训练（epoch {epoch})")
            break

    # 恢复最佳权重
    model.load_state_dict(best_state)
    print(f"[训练] 完成，最佳验证损失: {best_loss:.6e}")
    return model, train_losses, val_losses


# ============================================================================
# 5. 评估
# ============================================================================
def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    scaler_Y: StandardScaler,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """反归一化后返回 (true_vals, pred_vals)，形状均为 (N, 5)。"""
    model.eval()
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []

    with torch.no_grad():
        for Xb, Yb in val_loader:
            Xb = Xb.to(device)
            pred_scaled = model(Xb).cpu().numpy()
            all_pred.append(pred_scaled)
            all_true.append(Yb.numpy())

    pred_scaled = np.concatenate(all_pred, axis=0)
    true_scaled = np.concatenate(all_true, axis=0)

    pred = scaler_Y.inverse_transform(pred_scaled)
    true = scaler_Y.inverse_transform(true_scaled)
    return true, pred


def print_metrics(true: np.ndarray, pred: np.ndarray, param_names: list[str]) -> None:
    """打印每个参数的 MAE / RMSE / R² 表格。"""
    print()
    print(f"{'参数':<8} {'MAE':>12} {'RMSE':>12} {'R²':>12}")
    print("-" * 48)
    for i, name in enumerate(param_names):
        y_true = true[:, i]
        y_pred = pred[:, i]
        errors = y_true - y_pred
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))
        ss_res = np.sum(errors**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        print(f"{name:<8} {mae:>12.4f} {rmse:>12.4f} {r2:>12.4f}")
    print()


# ============================================================================
# 6. 绘图
# ============================================================================
def plot_loss_curve(
    train_losses: list[float],
    val_losses: list[float],
    save_path: str = "loss_curve.png",
) -> None:
    """保存训练/验证损失曲线。"""
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    plt.semilogy(epochs, train_losses, label="Train Loss")
    plt.semilogy(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss (log scale)")
    plt.title("Training and Validation Loss Curve")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[绘图] 损失曲线已保存: {save_path}")


# ============================================================================
# 7. 保存 / 加载
# ============================================================================
def save_artifacts(
    model: nn.Module,
    scaler_X: StandardScaler,
    scaler_Y: StandardScaler,
    true: np.ndarray,
    pred: np.ndarray,
    cfg: dict,
) -> None:
    """保存模型权重、归一化器和验证集预测。"""
    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # 模型检查点
    ckpt_path = save_dir / "best_lpm_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scaler_X": scaler_X,
            "scaler_Y": scaler_Y,
        },
        ckpt_path,
    )
    print(f"[保存] 模型权重: {ckpt_path}")

    # 验证集预测
    npz_path = save_dir / "test_predictions.npz"
    np.savez(
        npz_path,
        true_values=true,
        predicted_values=pred,
        param_names=PARAM_NAMES,
    )
    print(f"[保存] 预测结果: {npz_path}")


# ============================================================================
# 8. 主流程
# ============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="MLP LPM 参数替代模型训练")
    parser.add_argument("--mode", choices=["test", "full"], default=None, help="运行模式")
    parser.add_argument("--mat-path", default=None, help="synthetic_dataset.mat 路径")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", "--learning-rate", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    args = parser.parse_args()

    # 命令行参数覆盖 CFG
    for key, arg_val in {
        "mode": args.mode,
        "mat_path": args.mat_path,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "max_epochs": args.epochs,
        "seed": args.seed,
        "device": args.device,
    }.items():
        if arg_val is not None:
            CFG[key] = arg_val

    # 设备
    if CFG["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(CFG["device"])
    print(f"[设备] {device}")

    # 随机种子
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])

    # ---- 加载数据 ----
    X, Y = load_data(CFG)

    # ---- 预处理 ----
    train_loader, val_loader, scaler_X, scaler_Y = prepare_dataloaders(X, Y, CFG)

    # ---- 构建模型 ----
    model = MLPRegressor(
        in_dim=X.shape[1],
        out_dim=Y.shape[1],
        hidden_dims=CFG["hidden_dims"],
        dropout=CFG["dropout"],
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[模型] MLP({X.shape[1]} -> {CFG['hidden_dims']} -> {Y.shape[1]}), 参数量: {total_params:,}")
    print(model)

    # ---- 训练 ----
    model, train_losses, val_losses = train_model(model, train_loader, val_loader, CFG, device)

    # ---- 评估 ----
    true_vals, pred_vals = evaluate_model(model, val_loader, scaler_Y, device)
    print_metrics(true_vals, pred_vals, PARAM_NAMES)

    # ---- 绘图 ----
    plot_loss_curve(train_losses, val_losses, save_path=os.path.join(CFG["save_dir"], "loss_curve.png"))

    # ---- 保存 ----
    save_artifacts(model, scaler_X, scaler_Y, true_vals, pred_vals, CFG)

    print("[完成] 所有任务执行完毕。")


if __name__ == "__main__":
    main()
