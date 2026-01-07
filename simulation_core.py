import torch
import torch.nn as nn
import mesa
import random

# TCN Model
class TCN(nn.Module):
    def __init__(self, input_size=6, num_channels=[128, 128, 128], kernel_size=7, output_size=1):
        super(TCN, self).__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation = 2 ** i
            layers += [
                nn.Conv1d(input_size if i == 0 else num_channels[i-1], num_channels[i],
                          kernel_size, padding=(kernel_size-1)*dilation, dilation=dilation),
                nn.ReLU(),
                nn.BatchNorm1d(num_channels[i]),
                nn.Dropout(0.2)
            ]
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.tcn(x)  # [batch_size, num_channels[-1], seq_len]
        x = x[:, :, -1]  # [batch_size, num_channels[-1]]
        x = self.fc(x)   # [batch_size, output_size=1]
        x = self.sigmoid(x).squeeze(-1)  # [batch_size]
        return x

class RetroAgent(mesa.Agent):
    def __init__(self, unique_id, model, allow_collisions=False):
        super().__init__(unique_id, model)
        self.allow_collisions = allow_collisions

    def get_relative_positions(self):
        others = [agent for agent in self.model.schedule.agents if agent.unique_id != self.unique_id]
        rel_pos = []
        for other in others:
            dx = other.pos[0] - self.pos[0]
            dy = other.pos[1] - self.pos[1]
            rel_pos.extend([dx, dy])
        # Ensure exactly 10 values (for 5 other agents max padding)
        while len(rel_pos) < 10:
             rel_pos.extend([0.0, 0.0])
        return rel_pos[:10]

    def get_new_position(self, move):
        x, y = self.pos
        if move == 'up' and y < self.model.grid.height - 1: return (x, y + 1)
        if move == 'down' and y > 0: return (x, y - 1)
        if move == 'left' and x > 0: return (x - 1, y)
        if move == 'right' and x < self.model.grid.width - 1: return (x + 1, y)
        if move == 'stay': return (x, y)
        return (x, y)

    def step(self):
        apply_retro = (self.unique_id == 0 and self.model.tcn is not None)
        
        if apply_retro:
            max_val = 5.0
            pos_list = list(self.pos)
            rel_pos = self.get_relative_positions()
            current_features = [p / max_val for p in pos_list] + [r / max_val for r in rel_pos]
            input_seq = [current_features] * 5 
            input_tensor = torch.tensor([input_seq], dtype=torch.float32).transpose(1, 2)
            
            self.model.tcn.eval()
            with torch.no_grad():
                collision_prob = self.model.tcn(input_tensor).item()
            
            others_pos = [agent.pos for agent in self.model.schedule.agents if agent.unique_id != self.unique_id]
            moves = ['up', 'down', 'left', 'right', 'stay']
            
            if collision_prob > 0.5:
                # Avoid collision
                safe_moves = [m for m in moves if self.get_new_position(m) not in others_pos]
                move = random.choice(safe_moves) if safe_moves else 'stay'
            else:
                move = random.choice(moves)
        else:
            others_pos = [agent.pos for agent in self.model.schedule.agents if agent.unique_id != self.unique_id]
            moves = ['up', 'down', 'left', 'right', 'stay']
            
            if not self.allow_collisions:
                safe_moves = [m for m in moves if self.get_new_position(m) not in others_pos]
                move = random.choice(safe_moves) if safe_moves else 'stay'
            else:
                move = random.choice(moves)
        
        new_pos = self.get_new_position(move)
        if self.model.grid.is_cell_empty(new_pos) or self.model.allow_collisions or new_pos == self.pos:
            self.model.grid.move_agent(self, new_pos)

class RetroModel(mesa.Model):
    def __init__(self, allow_collisions=False, tcn_path=None, width=5, height=5, num_agents=3):
        super().__init__()
        self.grid = mesa.space.MultiGrid(width, height, torus=False)
        self.schedule = mesa.time.RandomActivation(self)
        self.allow_collisions = allow_collisions
        self.tcn = None
        
        if tcn_path:
            # Matches training config from notebook for 3 agents 5x5
            self.tcn = TCN(input_size=12, num_channels=[64, 64, 32], kernel_size=5, output_size=1)
            try:
                self.tcn.load_state_dict(torch.load(tcn_path, map_location=torch.device('cpu')))
            except Exception as e:
                print(f"Warning: Could not load TCN weights: {e}")
                
        for i in range(num_agents):
            agent = RetroAgent(i, self, allow_collisions)
            while True:
                pos = (random.randint(0, width-1), random.randint(0, height-1))
                if self.grid.is_cell_empty(pos):
                    break
            self.grid.place_agent(agent, pos)
            self.schedule.add(agent)

    def step(self):
        self.schedule.step()
