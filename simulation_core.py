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
            # Matches causal convolution structure from notebook
            padding = (kernel_size - 1) * dilation
            layers.append(nn.Conv1d(input_size if i == 0 else num_channels[i-1], num_channels[i],
                                    kernel_size, padding=padding, dilation=dilation))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            if padding > 0:
                layers.append(nn.ConstantPad1d((-padding, 0), 0))
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_size)
        self.output_size = output_size

    def forward(self, x):
        x = self.tcn(x)  # [batch_size, num_channels[-1], seq_len]
        x = x[:, :, -1]  # [batch_size, num_channels[-1]]
        x = self.fc(x)   # [batch_size, output_size]
        if self.output_size == 1:
            return torch.sigmoid(x).squeeze(-1)
        return x  # Return logits for multi-class

class RetroAgent(mesa.Agent):
    def __init__(self, model, allow_collisions=False):
        super().__init__(model)
        self.allow_collisions = allow_collisions
        self.history = []
        self.seq_len = 5

    def get_features(self):
        # Current state features
        max_x = float(self.model.grid.width)
        max_y = float(self.model.grid.height)

        pos_x = self.pos[0] / max_x
        pos_y = self.pos[1] / max_y

        others = [agent for agent in self.model.agents if agent.unique_id != self.unique_id]
        rel_pos = []
        for other in others:
            dx = (other.pos[0] - self.pos[0]) / max_x
            dy = (other.pos[1] - self.pos[1]) / max_y
            rel_pos.extend([dx, dy])

        # Pad features to match TCN input size
        input_size = self.model.tcn.tcn[0].in_channels if self.model.tcn else 12
        while len(rel_pos) < (input_size - 2):
             rel_pos.extend([0.0, 0.0])

        return ([pos_x, pos_y] + rel_pos)[:input_size]

    def update_history(self):
        self.history.append(self.get_features())
        if len(self.history) > self.seq_len:
            self.history.pop(0)

    def get_new_position(self, move):
        x, y = self.pos
        if move == 'up' and y < self.model.grid.height - 1: return (x, y + 1)
        if move == 'down' and y > 0: return (x, y - 1)
        if move == 'left' and x > 0: return (x - 1, y)
        if move == 'right' and x < self.model.grid.width - 1: return (x + 1, y)
        return (x, y)

    def step(self):
        self.update_history()

        # All agents attempt to use retrocausality if model has TCN
        apply_retro = (self.model.tcn is not None and len(self.history) == self.seq_len)

        moves = ['up', 'down', 'left', 'right', 'stay']
        others_pos = [agent.pos for agent in self.model.agents if agent.unique_id != self.unique_id]
        
        if apply_retro:
            # Use actual history for prediction
            input_tensor = torch.tensor([self.history], dtype=torch.float32).transpose(1, 2)
            
            self.model.tcn.eval()
            with torch.no_grad():
                output = self.model.tcn(input_tensor)
            
            if self.model.tcn.output_size == 1:
                # Collision prediction model
                collision_prob = output.item()
                if collision_prob > 0.5:
                    # Changing the future: Avoid predicted collision
                    safe_moves = [m for m in moves if self.get_new_position(m) not in others_pos]
                    move = random.choice(safe_moves) if safe_moves else 'stay'
                else:
                    move = random.choice(moves)
            else:
                # Move prediction model (multi-class)
                predicted_move_idx = torch.argmax(output, dim=1).item()
                # In 30-agent case, the model predicts the likely next move.
                # Retrocausal agents "change" their move if they want to deviate from predicted path?
                # Or they follow it but avoid collisions.
                # For "scientific accuracy," let's say they use the prediction to stay safe.
                move = moves[predicted_move_idx % 5]
                if not self.allow_collisions and self.get_new_position(move) in others_pos:
                    safe_moves = [m for m in moves if self.get_new_position(m) not in others_pos]
                    move = random.choice(safe_moves) if safe_moves else 'stay'
        else:
            if not self.allow_collisions:
                safe_moves = [m for m in moves if self.get_new_position(m) not in others_pos]
                move = random.choice(safe_moves) if safe_moves else 'stay'
            else:
                move = random.choice(moves)
        
        new_pos = self.get_new_position(move)
        if self.model.grid.is_cell_empty(new_pos) or self.model.allow_collisions or new_pos == self.pos:
            self.model.grid.move_agent(self, new_pos)

class RetroModel(mesa.Model):
    def __init__(self, allow_collisions=False, tcn_path=None, width=10, height=10, num_agents=30):
        super().__init__()
        self.grid = mesa.space.MultiGrid(width, height, torus=False)
        self.allow_collisions = allow_collisions
        self.tcn = None
        self.agent0_collisions = 0
        self.total_collisions = 0
        self.history_data = []
        
        if tcn_path:
            # Detect architecture from state dict if possible, or use sensible defaults
            try:
                sd = torch.load(tcn_path, map_location=torch.device('cpu'))
                input_size = sd['tcn.0.weight'].shape[1]
                output_size = sd['fc.weight'].shape[0]
                # Infer hidden channels (approximate)
                num_channels = [64, 64, 32] if input_size == 12 else [128, 128, 128]
                kernel_size = 5 if input_size == 12 else 7

                self.tcn = TCN(input_size=input_size, num_channels=num_channels, kernel_size=kernel_size, output_size=output_size)
                self.tcn.load_state_dict(sd)
                self.tcn.eval()
            except Exception as e:
                print(f"Warning: Could not load TCN weights or infer architecture: {e}")
                # Fallback
                self.tcn = TCN(input_size=12, num_channels=[64, 64, 32], kernel_size=5, output_size=1)
                
        for i in range(num_agents):
            agent = RetroAgent(self, allow_collisions)
            agent.unique_id = i
            # Find a random empty position
            attempts = 0
            while attempts < 100:
                pos = (self.random.randint(0, width-1), self.random.randint(0, height-1))
                if self.grid.is_cell_empty(pos):
                    break
                attempts += 1
            else:
                # If grid is very full, just place it anywhere
                pos = (self.random.randint(0, width-1), self.random.randint(0, height-1))

            self.grid.place_agent(agent, pos)

    def step(self):
        self.agents.shuffle().do("step")
        self.steps += 1
        self.track_collisions()
        self.log_data()

    def track_collisions(self):
        agent_positions = [agent.pos for agent in self.agents]
        unique_positions = set(agent_positions)
        collisions_this_step = len(agent_positions) - len(unique_positions)
        self.total_collisions += collisions_this_step

        # Track agent 0 specifically
        agent0 = [a for a in self.agents if a.unique_id == 0][0]
        others_pos = [a.pos for a in self.agents if a.unique_id != 0]
        if agent0.pos in others_pos:
            self.agent0_collisions += 1

    def log_data(self):
        step_info = {
            'step': self.steps,
            'total_collisions': self.total_collisions,
            'agent0_collisions': self.agent0_collisions,
            'positions': {a.unique_id: a.pos for a in self.agents}
        }
        self.history_data.append(step_info)
