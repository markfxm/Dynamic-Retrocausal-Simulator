import torch
import torch.nn as nn
import numpy as np
import pickle
import pathlib
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from simulation_core import TCN
from train_model import prepare_training_data

def evaluate_model(model, X, y, criterion):
    dataset = TensorDataset(torch.FloatTensor(X).permute(0, 2, 1), torch.FloatTensor(y))
    loader = DataLoader(dataset, batch_size=64)
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
            predictions.extend(preds.numpy())
            
    return total_loss / total, correct / total, np.array(predictions)

def run_analysis():
    print("--- DEEP MODEL ANALYSIS ---")
    data_path = pathlib.Path('MixedData/abm_mixed_3agents_5x5_collisions.pkl')
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    X, y = prepare_training_data(data)
    
    model = TCN(input_size=12, num_channels=[64, 64, 32], kernel_size=5, output_size=1)
    try:
        model.load_state_dict(torch.load('MixedData/tcn_model.pt'))
        print("Model loaded successfully.")
    except:
        print("Could not load model weights. Please run train_model.py first.")
        return

    criterion = nn.BCEWithLogitsLoss()
    loss, acc, preds = evaluate_model(model, X, y, criterion)
    print(f"\n[Baseline] Accuracy: {acc:.4f}")
    
    print("\n--- 1. Failure Mode Analysis (Confusion Matrix) ---")
    cm = confusion_matrix(y, preds)
    tn, fp, fn, tp = cm.ravel()
    print(f"True Negatives (Safe, Predicted Safe): {tn}")
    print(f"False Positives (Safe, Predicted Collision): {fp} (Paranoia)")
    print(f"False Negatives (Collision, Predicted Safe): {fn} (Blind Spots - DANGEROUS)")
    print(f"True Positives (Collision, Predicted Collision): {tp}")
    
    print("\n--- 2. Feature Importance (Permutation) ---")
    feature_names = ['Pos X', 'Pos Y'] + [f'Rel Pos {i}' for i in range(10)]
    baseline_acc = acc
    importances = {}
    for i in range(12):
        X_permuted = X.copy()
        np.random.shuffle(X_permuted[:, :, i])
        _, perm_acc, _ = evaluate_model(model, X_permuted, y, criterion)
        importances[feature_names[i]] = baseline_acc - perm_acc
        
    sorted_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    for name, imp in sorted_feats:
        print(f"{name}: {imp:.4f} impact")
        
    print("\n--- 3. Temporal Ablation (Does History Matter?) ---")
    X_no_hist = X.copy()
    X_no_hist[:, :4, :] = 0
    _, acc_no_hist, _ = evaluate_model(model, X_no_hist, y, criterion)
    print(f"Accuracy with NO History (Last step only): {acc_no_hist:.4f} (Drop: {baseline_acc - acc_no_hist:.4f})")
    
    X_only_hist = X.copy()
    X_only_hist[:, 4, :] = 0
    _, acc_only_hist, _ = evaluate_model(model, X_only_hist, y, criterion)
    print(f"Accuracy with ONLY History (Blind to current): {acc_only_hist:.4f} (Drop: {baseline_acc - acc_only_hist:.4f})")

if __name__ == "__main__":
    run_analysis()
