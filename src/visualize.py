import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import numpy as np
from pathlib import Path

# ================== 配置导入 ==================
from config import (
    GRID_WIDTH, GRID_HEIGHT, NUM_AGENTS,
    TCN_MODEL_PATH, RESULTS_DIR, STEPS
)
from src.model import RetroModel

# 确保结果目录存在
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def create_animation(steps=20, output_file=None):
    """
    生成随机模型 vs Retrocausal (TCN) 模型的对比动画
    清晰展示 Agent 0 使用 TCN 后碰撞大幅减少的效果
    """
    if output_file is None:
        output_file = RESULTS_DIR / "simulation_comparison.gif"

    # Case 1: 纯随机模型（无规则，允许碰撞）
    model_random = RetroModel(
        allow_collisions=True,
        tcn_path=None,                    # 明确不加载 TCN
        width=GRID_WIDTH,
        height=GRID_HEIGHT,
        num_agents=NUM_AGENTS
    )

    # Case 2: Retrocausal 模型（Agent 0 使用 TCN 预测避免碰撞）
    model_retro = RetroModel(
        allow_collisions=True,
        tcn_path=TCN_MODEL_PATH,
        width=GRID_WIDTH,
        height=GRID_HEIGHT,
        num_agents=NUM_AGENTS
    )

    # 初始化碰撞计数器（避免 AttributeError）
    model_random.collision_count = 0
    model_retro.collision_count = 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    def update(frame):
        # 执行一步
        model_random.step()
        model_retro.step()

        # 清空画布
        ax1.clear()
        ax2.clear()

        # ================== 左侧：Random Agents ==================
        ax1.set_xlim(-0.5, GRID_WIDTH - 0.5)
        ax1.set_ylim(-0.5, GRID_HEIGHT - 0.5)
        ax1.set_xticks(range(GRID_WIDTH))
        ax1.set_yticks(range(GRID_HEIGHT))
        ax1.grid(True, alpha=0.6)
        ax1.set_title(f"Random Agents (No Rule)\n"
                      f"Step {frame} | Collisions: {model_random.collision_count}")

        # ================== 右侧：Retrocausal Agents ==================
        ax2.set_xlim(-0.5, GRID_WIDTH - 0.5)
        ax2.set_ylim(-0.5, GRID_HEIGHT - 0.5)
        ax2.set_xticks(range(GRID_WIDTH))
        ax2.set_yticks(range(GRID_HEIGHT))
        ax2.grid(True, alpha=0.6)
        ax2.set_title(f"Retrocausal Agents (TCN Rule)\n"
                      f"Step {frame} | Collisions: {model_retro.collision_count}")

        # 绘制两个模型的 agents
        plot_agents(model_random, ax1, is_retro=False)
        plot_agents(model_retro, ax2, is_retro=True)

        # 实时统计碰撞（只统计 Agent 0 是否在碰撞位置）
        update_collision_count(model_random)
        update_collision_count(model_retro)

    def plot_agents(model, ax, is_retro=False):
        pos_count = {}
        for agent in model.schedule.agents:
            pos = agent.pos
            pos_count[pos] = pos_count.get(pos, 0) + 1

        for agent in model.schedule.agents:
            x, y = agent.pos
            if agent.unique_id == 0:
                color = 'blue' if not is_retro else 'purple'   # Agent 0 高亮
            else:
                color = 'green'

            # 碰撞显示为红色
            if pos_count.get(pos, 0) > 1:
                color = 'red'

            ax.plot(x, y, 'o', markersize=22, color=color, alpha=0.85)
            ax.text(x, y, str(agent.unique_id),
                    color='white', ha='center', va='center',
                    fontweight='bold', fontsize=10)

    def update_collision_count(model):
        """统计当前是否有碰撞（任意两个 agent 在同一格）"""
        pos_count = {}
        for agent in model.schedule.agents:
            pos_count[agent.pos] = pos_count.get(agent.pos, 0) + 1

        # 只要有格子人数 >1，就算一次碰撞事件（简单计数）
        collisions_this_step = sum(1 for count in pos_count.values() if count > 1)
        model.collision_count += collisions_this_step

    print(f"Generating animation ({steps} steps)...")
    ani = animation.FuncAnimation(fig, update, frames=steps, repeat=False, interval=300)

    try:
        ani.save(output_file, writer='pillow', fps=2)
        print(f"✅ Animation saved to: {output_file}")
        print(f"   Random collisions: {model_random.collision_count}")
        print(f"   Retrocausal collisions: {model_retro.collision_count}")
    except Exception as e:
        print(f"❌ Error saving animation: {e}")


if __name__ == "__main__":
    # 你可以改 steps=7 来匹配你的 mini 项目主循环
    create_animation(steps=STEPS * 2)   # 默认跑 14 步，让差异更明显