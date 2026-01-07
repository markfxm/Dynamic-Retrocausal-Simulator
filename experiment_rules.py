import mesa
import random
import pickle
import torch
import numpy as np
import pathlib
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
from simulation_core import RetroAgent, RetroModel, TCN
from train_model import prepare_training_data

# 1. Define Deterministic Agent
class DeterministicAgent(RetroAgent):
    def step(self):
        # Rule: Move Clockwise (Right -> Down -> Left -> Up) based on position
        # Or simpler: Prefer R, if not D, if not L, if not U
        # This makes the path highly predictable (cycles)
        
        moves_priority = ['right', 'down', 'left', 'up']
        
        # Override random choice
        chosen_move = 'stay'
        for move in moves_priority:
            new_pos = self.get_new_position(move)
            if self.model.grid.is_cell_empty(new_pos) or self.model.allow_collisions:
                # Still check boundaries which get_new_position handles by returning same pos
                if new_pos != self.pos:
                     chosen_move = move
                     break
        
        # If blocked or at edge (and edge move is invalid), might stay. 
        # But this is deterministic sequence.
        
        new_pos = self.get_new_position(chosen_move)
        if self.model.grid.is_cell_empty(new_pos) or self.model.allow_collisions or new_pos == self.pos:
            self.model.grid.move_agent(self, new_pos)

# 2. Collect Data
def collect_deterministic_data(num_sims=500, steps=10):
    all_data = []
    print(f"Collecting data from {num_sims} simulations with Deterministic Agents...")
    
    for _ in range(num_sims):
        model = RetroModel(allow_collisions=True, width=5, height=5, num_agents=3)
        # Replace agents with Deterministic ones
        # (A bit of a hack, but easier than rewriting Model)
        # Reset schedule and grid
        model.grid = mesa.space.MultiGrid(5, 5, torus=False)
        model.schedule = mesa.time.RandomActivation(model)
        for i in range(3):
            agent = DeterministicAgent(i, model, allow_collisions=True)
            while True:
                pos = (random.randint(0, 4), random.randint(0, 4))
                if model.grid.is_cell_empty(pos):
                    break
            model.grid.place_agent(agent, pos)
            model.schedule.add(agent)
            
        data = []
        for _ in range(steps):
             model.step()
             # Log data similar to notebook
             agent0 = [a for a in model.schedule.agents if a.unique_id == 0][0]
             others = [a for a in model.schedule.agents if a.unique_id != 0]
             
             # Calculate collision for agent 0
             # Note: collision logic in notebook was a bit specific: 
             # "collision = next_pos in others_pos" calculated BEFORE move in model.step
             # Here we are post-step.
             # Let's assume for training it checks if CURRENT pos collision? No, notebook used `collision` flag.
             # We need to capture if agent 0 collided. 
             # Since we replaced step logic, we didn't track collision count explicitly in Agent.step.
             # But if allow_collisions=True, multiple agents can be in same cell.
             
             is_collision = False
             current_pos = agent0.pos
             # specific check if sharing cell with others
             for o in others:
                 if o.pos == current_pos:
                     is_collision = True
                     break
             
             for agent in model.schedule.agents:
                data.append({
                    'agent_id': agent.unique_id,
                    'step': model.schedule.steps,
                    'pos': agent.pos,
                    'rel_pos': agent.get_relative_positions(),
                    'collision': is_collision and agent.unique_id == 0
                })
        all_data.extend(data)
        
    return all_data

# 3. Train and Compare
def run_experiment():
    # A. Baseline (Random - we already trained it, but let's re-eval or just use the number 0.97)
    print("Running Experiment: Deterministic Rules vs Random")
    
    # Generate Deterministic Data
    det_data = collect_deterministic_data()
    X_det, y_det = prepare_training_data(det_data)
    print(f"Deterministic Data: {len(y_det)} samples. Collision Rate: {y_det.mean():.4f}")
    
    # Train TCN on Deterministic Data
    X_train, X_val, y_train, y_val = train_test_split(X_det, y_det, test_size=0.2, random_state=42)
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train).permute(0, 2, 1), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val).permute(0, 2, 1), torch.FloatTensor(y_val))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    model = TCN(input_size=12, num_channels=[64, 64, 32], kernel_size=5, output_size=1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    best_acc = 0
    print("Training on Deterministic Data...")
    for epoch in range(20):
        model.train()
        for X_b, y_b in train_loader:
             optimizer.zero_grad()
             loss = criterion(model(X_b), y_b)
             loss.backward()
             optimizer.step()
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
             for X_b, y_b in val_loader:
                 preds = (model(X_b) > 0.5).float()
                 correct += (preds == y_b).sum().item()
                 total += y_b.size(0)
        acc = correct / total
        if acc > best_acc: best_acc = acc
        print(f"Epoch {epoch+1}: Acc {acc:.4f}")
        
    print(f"\nFinal Result:")
    print(f"Deterministic Rule Accuracy: {best_acc:.4f}")
    print(f"Baseline (Random) Accuracy: ~0.9728 (from previous run)")
    
    if best_acc > 0.9728:
        print(">> Hypothesis SUPPORTED: More rules (determinism) -> Higher accuracy.")
    elif best_acc > 0.90:
        print(">> Hypothesis PLAUSIBLE: Accuracy is very high, comparable to baseline.")
    else:
        print(">> Hypothesis UNCLEAR.")

if __name__ == "__main__":
    run_experiment()
