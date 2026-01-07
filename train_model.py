import torch
import torch.nn as nn
import pickle
import numpy as np
import pathlib
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from simulation_core import TCN

def prepare_training_data(data, seq_len=5):
    # Filter for agent 0 as the "retrocausal" agent
    agent0_data = [d for d in data if d['agent_id'] == 0]
    X, y = [], []
    
    # Needs 5 steps of history to predict next collision
    for i in range(len(agent0_data) - seq_len):
        sequence = []
        for j in range(i, i + seq_len):
            max_x = 5.0 # Grid size
            max_y = 5.0
            
            # Notebook data prep:
            pos_x = agent0_data[j]['pos'][0] / max_x
            pos_y = agent0_data[j]['pos'][1] / max_y
            
            # Process relative positions
            rel_pos_raw = agent0_data[j]['rel_pos']
            # Notebook had specific normalization: x/max_x if i%2==0 else x/max_y
            rel_pos = [x / max_x if k % 2 == 0 else x / max_y for k, x in enumerate(rel_pos_raw)]
            
            # Pad to 10 relative pos values if needed (total 12 features)
            while len(rel_pos) < 10:
                rel_pos.extend([0.0, 0.0])
            rel_pos = rel_pos[:10]
            
            features = [pos_x, pos_y] + rel_pos
            sequence.append(features)
        
        X.append(sequence)
        # Target: Collision at the END of the sequence (or next step? Notebook says: agent0_data[i + seq_len]['collision'])
        # Actually in notebook it's: y.append(agent0_data[i + seq_len]['collision'])
        # Which effectively is predicting if the state AFTER the sequence is a collision? 
        # Or is 'collision' in the data marking if the step *resulted* in a collision?
        # Let's assume notebook logic is correct: predicting collision at t+5 based on t..t+4
        y.append(float(agent0_data[i + seq_len]['collision']))
        
    return np.array(X), np.array(y)

def train():
    # Load Data
    data_path = pathlib.Path('MixedData/abm_mixed_3agents_5x5_collisions.pkl')
    if not data_path.exists():
        print(f"Error: {data_path} not found.")
        return

    with open(data_path, 'rb') as f:
        data = pickle.load(f)
        
    print(f"Loaded {len(data)} data points.")
    X, y = prepare_training_data(data)
    print(f"Training Data: X {X.shape}, y {y.shape}")
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # To Tensor
    # Model expects [batch, channels, seq_len] -> [batch, 12, 5]
    # X is [samples, seq_len, features] -> transpose to [samples, features, seq_len]
    train_dataset = TensorDataset(torch.FloatTensor(X_train).permute(0, 2, 1), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val).permute(0, 2, 1), torch.FloatTensor(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Init Model
    # Input size 12 (2 pos + 10 rel)
    model = TCN(input_size=12, num_channels=[64, 64, 32], kernel_size=5, output_size=1)
    
    # Weighted Loss for Imbalance
    if y.mean() > 0:
        pos_weight = torch.tensor([(1 - y.mean()) / y.mean()])
    else:
        pos_weight = torch.tensor([1.0])
        
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train
    best_val_acc = 0
    patience = 10
    counter = 0
    
    print("Starting training...")
    for epoch in range(50): # 50 epochs
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        val_correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                predicted = (outputs > 0.5).float()
                total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()
        
        val_acc = val_correct / total
        print(f"Epoch {epoch+1}: Val Acc {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'MixedData/tcn_model.pt')
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping.")
                break
                
    print(f"Training complete. Best Acc: {best_val_acc:.4f}")
    print("Model saved to MixedData/tcn_model.pt")

if __name__ == "__main__":
    train()
