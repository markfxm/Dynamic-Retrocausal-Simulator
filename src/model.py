import pathlib

from config import *
import torch
import torch.nn as nn
import mesa
import random
from config import MAX_VAL

# TCN Model
class TCN(nn.Module):
    def __init__(self, input_size=12, num_channels=[128, 128, 128], kernel_size=7, output_size=1):
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
        self.history = []

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
        moves = ['up', 'down', 'left', 'right', 'stay']
        others_pos = [agent.pos for agent in self.model.schedule.agents if agent.unique_id != self.unique_id]

        if apply_retro and self.model.tcn is not None:
            # === 关键升级：构造真实历史序列 ===
            max_val = MAX_VAL
            current_features = []
            pos_list = list(self.pos)
            rel_pos = self.get_relative_positions()
            current_features = [p / max_val for p in pos_list] + [r / max_val for r in rel_pos]

            # 更新历史（保持最近 SEQ_LEN 步）
            self.history.append(current_features)
            if len(self.history) > SEQ_LEN:
                self.history.pop(0)

            # 如果历史不够长，用当前帧填充
            seq = self.history + [current_features] * (SEQ_LEN - len(self.history))
            
            input_tensor = torch.tensor([seq], dtype=torch.float32).permute(0, 2, 1).to('cpu')  # [1, 12, SEQ_LEN]

            self.model.tcn.eval()
            with torch.no_grad():
                collision_prob = self.model.tcn(input_tensor).item()

            # 决策逻辑（加强版）
            if collision_prob > 0.5:                     # 预测会碰撞 → 强避让 + center bias
                safe_moves = [m for m in moves if self.get_new_position(m) not in others_pos]
                if safe_moves:
                    center = (GRID_WIDTH // 2, GRID_HEIGHT // 2)
                    safe_moves.sort(key=lambda m: 
                        abs(self.get_new_position(m)[0] - center[0]) + 
                        abs(self.get_new_position(m)[1] - center[1]))
                    move = safe_moves[0]
                else:
                    move = 'stay'
            else:
                # 安全时：随机 + 35% 趋向中心（增加确定性）
                if random.random() < 0.35:
                    center_moves = [m for m in moves if self.get_new_position(m) not in others_pos]
                    if center_moves:
                        center = (GRID_WIDTH // 2, GRID_HEIGHT // 2)
                        center_moves.sort(key=lambda m: 
                            abs(self.get_new_position(m)[0] - center[0]) + 
                            abs(self.get_new_position(m)[1] - center[1]))
                        move = center_moves[0]
                    else:
                        move = random.choice(moves)
                else:
                    move = random.choice(moves)
        else:
            # 非 Agent 0 或无 TCN：纯随机
            if not self.allow_collisions:
                safe_moves = [m for m in moves if self.get_new_position(m) not in others_pos]
                move = random.choice(safe_moves) if safe_moves else 'stay'
            else:
                move = random.choice(moves)

        new_pos = self.get_new_position(move)
        if self.model.grid.is_cell_empty(new_pos) or self.model.allow_collisions or new_pos == self.pos:
            self.model.grid.move_agent(self, new_pos)


class RetroModel(mesa.Model):
    def __init__(self, 
                 allow_collisions=ALLOW_COLLISIONS, 
                 tcn_path=None, 
                 width=GRID_WIDTH, 
                 height=GRID_HEIGHT, 
                 num_agents=NUM_AGENTS):
        super().__init__()
        self.grid = mesa.space.MultiGrid(width, height, torus=False)
        self.schedule = mesa.time.RandomActivation(self)
        self.allow_collisions = allow_collisions
        self.tcn = None
        
        # 创建 agents 并放置到网格上（随机位置）
        for i in range(num_agents):
            agent = RetroAgent(i, self, allow_collisions=allow_collisions)
            # 随机放置 agent 在网格上
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            self.grid.place_agent(agent, (x, y))
            self.schedule.add(agent)
        
        # 只在明确传入有效路径时才尝试加载 TCN（数据生成时不要加载）
        if tcn_path and pathlib.Path(tcn_path).exists():
            try:
                self.tcn = TCN(
                    input_size=TCN_INPUT_SIZE,
                    num_channels=TCN_NUM_CHANNELS,
                    kernel_size=TCN_KERNEL_SIZE,
                    output_size=TCN_OUTPUT_SIZE
                )
                state_dict = torch.load(tcn_path, map_location=torch.device('cpu'))
                self.tcn.load_state_dict(state_dict)
                self.tcn.to(torch.device('cpu'))
                self.tcn.eval()
                #print(f"✅ TCN 模型已加载到 CPU: {tcn_path}")
            except Exception as e:
                print(f"Warning: Could not load TCN: {e}")
                self.tcn = None

    def step(self):
        self.schedule.step()

    def run_simulation(self, steps=STEPS):
        """Run a single simulation and return collected data"""
        data = []
        collision_count = 0
        
        for step in range(steps):
            # Record current state
            for agent in list(self.schedule.agents):
                agent_id = agent.unique_id
                pos = agent.pos
                rel_pos = agent.get_relative_positions()

                others_pos = [a.pos for a in self.schedule.agents if a.unique_id != agent_id]
                collision = 1 if pos in others_pos else 0
                
                step_data = {
                    'agent_id': agent_id,
                    'pos': pos,
                    'rel_pos': rel_pos,
                    'collision': collision,
                    'step': step
                }
                data.append(step_data)

                if collision and agent_id == 0:
                    collision_count += 1

            # Execute step
            self.step()

        return data, collision_count
