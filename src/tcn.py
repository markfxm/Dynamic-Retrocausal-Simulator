import torch
import torch.nn as nn
import pickle
import numpy as np
import pathlib
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from config import *
from src.model import TCN   # ← 必须导入 TCN 类

def prepare_training_data(data, seq_len=SEQ_LEN):
    """兼容当前 generate_data 保存的结构：data 是 list of runs，每个 run 是 list of dicts"""
    X, y = [], []
    total_sequences = 0
    
    print(f"Processing {len(data)} runs for Agent 0...")

    for run_idx, run in enumerate(data):
        # run 可能是 list of dict，也可能是其他，先强制转为可迭代的 dict 列表
        if not isinstance(run, (list, tuple)):
            continue
            
        # 提取 Agent 0 的记录
        agent0_data = []
        for record in run:
            if isinstance(record, dict) and record.get('agent_id') == 0:
                agent0_data.append(record)
        
        # 按 step 排序
        agent0_data.sort(key=lambda x: x.get('step', 0))
        
        if len(agent0_data) < seq_len + 1:
            continue
            
        for i in range(len(agent0_data) - seq_len):
            sequence = []
            for j in range(i, i + seq_len):
                record = agent0_data[j]
                pos = record.get('pos', (0, 0))
                rel_pos_raw = record.get('rel_pos', [0.0] * 10)
                
                # 归一化
                pos_x = float(pos[0]) / GRID_WIDTH
                pos_y = float(pos[1]) / GRID_HEIGHT
                
                rel_pos = []
                for k, val in enumerate(rel_pos_raw):
                    rel_pos.append(float(val) / GRID_WIDTH if k % 2 == 0 else float(val) / GRID_HEIGHT)
                
                while len(rel_pos) < 10:
                    rel_pos.extend([0.0, 0.0])
                rel_pos = rel_pos[:10]
                
                features = [pos_x, pos_y] + rel_pos
                sequence.append(features)
            
            X.append(sequence)
            # 预测下一个 step 是否碰撞
            next_record = agent0_data[i + seq_len]
            y.append(float(next_record.get('collision', 0)))
            total_sequences += 1
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    print(f"Generated {len(X)} training sequences from Agent 0 (total_sequences={total_sequences})")
    if len(X) == 0:
        print("⚠️  Warning: No sequences generated. Data structure may still not match.")
    
    return X, y
def train():
    data_path = pathlib.Path(MIXED_DATA_PATH)
    if not data_path.exists():
        print(f"❌ Error: Data file not found at {data_path}")
        return

    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"✅ Loaded {len(data)} runs")
    X, y = prepare_training_data(data)
    if len(X) == 0:
        print("❌ No valid sequences generated.")
        return
    
    print(f"Training samples: {X.shape[0]:,} | Positive rate: {y.mean():.4f} ({int(y.sum())} collisions)")

    # 数据划分（使用 stratify 保持比例）
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    train_dataset = TensorDataset(torch.FloatTensor(X_train).permute(0, 2, 1), torch.FloatTensor(y_train))
    val_dataset   = TensorDataset(torch.FloatTensor(X_val).permute(0, 2, 1),   torch.FloatTensor(y_val))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = TCN(
        input_size=TCN_INPUT_SIZE,
        num_channels=TCN_NUM_CHANNELS,
        kernel_size=TCN_KERNEL_SIZE,
        output_size=TCN_OUTPUT_SIZE
    )

    # 强力不平衡处理
    pos_weight = torch.tensor([(1 - y.mean()) / max(y.mean(), 1e-6) * 3.0])  # 加大权重
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_val_acc = 0.0
    best_recall = 0.0
    counter = 0

    print("\n🚀 Starting TCN Training - Retrocausal Collision Predictor")
    print("-" * 85)
    print(f"{'Epoch':<5} {'Train Loss':<12} {'Val Acc':<10} {'Val Recall':<12} {'Best Acc':<10} {'Best Recall':<12} Status")
    print("-" * 85)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_dataset)

        # Validation
        model.eval()
        val_correct = val_total = val_tp = val_fn = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                preds = (outputs > 0.5).float()
                val_correct += (preds == y_batch).sum().item()
                val_total += y_batch.size(0)
                val_tp += ((preds == 1) & (y_batch == 1)).sum().item()
                val_fn += ((preds == 0) & (y_batch == 1)).sum().item()

        val_acc = val_correct / val_total
        val_recall = val_tp / (val_tp + val_fn) if (val_tp + val_fn) > 0 else 0.0

        scheduler.step(val_acc)

        status = ""
        improved = False
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            improved = True
        if val_recall > best_recall:
            best_recall = val_recall
            improved = True

        if improved:
            torch.save(model.state_dict(), TCN_MODEL_PATH)
            status = "★ Saved"
            counter = 0
        else:
            counter += 1
            if counter >= PATIENCE:
                status = "Early Stop"
                print("-" * 85)
                break

        print(f"{epoch+1:<5} {train_loss:<12.4f} {val_acc:<10.4f} {val_recall:<12.4f} "
              f"{best_val_acc:<10.4f} {best_recall:<12.4f} {status}")

    print("-" * 85)
    print(f"🎉 Training Completed!")
    print(f"   Best Accuracy : {best_val_acc:.4f}")
    print(f"   Best Recall   : {best_recall:.4f}")
    print(f"   Model saved to: {TCN_MODEL_PATH}")

if __name__ == "__main__":
    train()