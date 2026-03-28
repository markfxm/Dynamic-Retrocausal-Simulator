import torch
import torch.nn as nn
import numpy as np
import pickle
import pathlib
import gc
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

# 项目配置
from config import (
    MIXED_DATA_PATH, TCN_MODEL_PATH, TCN_INPUT_SIZE,
    TCN_NUM_CHANNELS, TCN_KERNEL_SIZE, TCN_OUTPUT_SIZE,
    SEQ_LEN, GRID_WIDTH, GRID_HEIGHT, device
)
from src.model import TCN
from src.tcn import prepare_training_data

# ====================== 强制 CPU（你的 RTX 5070 需要） ======================
device = torch.device('cpu')
print(f"⚠️ evaluate.py 已强制使用 CPU | device = {device}")

def evaluate_model(model, X, y, criterion):
    """评估模型，返回 loss, accuracy, predictions"""
    # 确保输入是 CPU
    dataset = TensorDataset(
        torch.FloatTensor(X).permute(0, 2, 1).to(device),
        torch.FloatTensor(y).to(device)
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    predictions = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item() * y_batch.size(0)
            
            preds = (outputs > 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
            predictions.extend(preds.cpu().numpy())
            
    return total_loss / total, correct / total, np.array(predictions)


def compare_collision_reduction(num_sims=100, steps=7, tcn_path=None):
    """你最关心的对比实验（已集成到 evaluate.py）"""
    from src.model import RetroModel
    import importlib
    importlib.reload(src.model)
    
    print(f"🚀 运行碰撞减少对比: {num_sims} 次模拟 × {steps} steps")
    
    coll_no = 0
    coll_yes = 0
    
    for i in range(num_sims):
        # 无 TCN（纯随机）
        model_no = RetroModel(allow_collisions=True, tcn_path=None,
                              width=GRID_WIDTH, height=GRID_HEIGHT, num_agents=3)
        _, c_no = model_no.run_simulation(steps=steps)
        coll_no += c_no
        
        # 有 TCN（真实历史序列）
        model_yes = RetroModel(allow_collisions=True, tcn_path=tcn_path,
                               width=GRID_WIDTH, height=GRID_HEIGHT, num_agents=3)
        _, c_yes = model_yes.run_simulation(steps=steps)
        coll_yes += c_yes
    
    avg_no = coll_no / num_sims
    avg_yes = coll_yes / num_sims
    reduction = (avg_no - avg_yes) / avg_no * 100 if avg_no > 0 else 0
    
    print(f"\n✅ 对比完成！")
    print(f"   无 TCN 规则平均碰撞 (Agent 0): {avg_no:.3f}")
    print(f"   有 TCN 规则平均碰撞 (Agent 0): {avg_yes:.3f}")
    print(f"   碰撞减少比例: {reduction:.1f}%")
    
    return avg_no, avg_yes, reduction


def run_analysis():
    print("=== DEEP MODEL ANALYSIS (Retrocausal TCN) ===")
    print(f"使用设备: {device}\n")
    
    # 加载数据
    data_path = pathlib.Path(MIXED_DATA_PATH)
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    X, y = prepare_training_data(data)
    print(f"加载数据完成: {X.shape} sequences, 正样本率 {y.mean():.4f}")
    
    # 加载模型（强制 CPU）
    model = TCN(
        input_size=TCN_INPUT_SIZE,
        num_channels=TCN_NUM_CHANNELS,
        kernel_size=TCN_KERNEL_SIZE,
        output_size=TCN_OUTPUT_SIZE
    )
    try:
        state_dict = torch.load(TCN_MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print(f"✅ 模型加载成功: {TCN_MODEL_PATH}")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("请确认已运行训练并生成 tcn_best.pt")
        return

    criterion = nn.BCEWithLogitsLoss()
    loss, acc, preds = evaluate_model(model, X, y, criterion)
    
    print(f"\n[Baseline] Accuracy: {acc:.4f} | Loss: {loss:.4f}")
    
    print("\n--- 1. Failure Mode Analysis (Confusion Matrix) ---")
    cm = confusion_matrix(y, preds)
    tn, fp, fn, tp = cm.ravel()
    print(f"True Negatives  (Safe → Safe)     : {tn}")
    print(f"False Positives (Safe → Collision): {fp} (Paranoia)")
    print(f"False Negatives (Collision → Safe): {fn} (DANGEROUS - Blind Spot)")
    print(f"True Positives  (Collision → Collision): {tp}")
    
    print("\n--- 2. Feature Importance (Permutation) ---")
    feature_names = ['Pos X', 'Pos Y'] + [f'Rel Pos {i}' for i in range(10)]
    baseline_acc = acc
    importances = {}
    for i in range(TCN_INPUT_SIZE):
        X_permuted = X.copy()
        np.random.shuffle(X_permuted[:, :, i])
        _, perm_acc, _ = evaluate_model(model, X_permuted, y, criterion)
        importances[feature_names[i]] = baseline_acc - perm_acc
    
    sorted_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    for name, imp in sorted_feats[:8]:   # 只显示前8个
        print(f"{name:12}: {imp:.4f} impact")
    
    print("\n--- 3. Temporal Ablation (History Matters?) ---")
    X_no_hist = X.copy()
    X_no_hist[:, :4, :] = 0   # 清空前4步历史
    _, acc_no_hist, _ = evaluate_model(model, X_no_hist, y, criterion)
    print(f"Only Last Step (No History) : {acc_no_hist:.4f} (Drop: {baseline_acc - acc_no_hist:.4f})")
    
    X_only_hist = X.copy()
    X_only_hist[:, -1, :] = 0   # 清空当前步（只看历史）
    _, acc_only_hist, _ = evaluate_model(model, X_only_hist, y, criterion)
    print(f"Only History (Blind to Current): {acc_only_hist:.4f} (Drop: {baseline_acc - acc_only_hist:.4f})")


if __name__ == "__main__":
    run_analysis()
    # 可选：直接运行对比
    # compare_collision_reduction(num_sims=100, steps=7, tcn_path=TCN_MODEL_PATH)