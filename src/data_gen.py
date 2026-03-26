import pickle
import random
from tqdm import tqdm
from config import *
from src.model import RetroModel

def generate_data(num_runs=300, steps_per_run=STEPS, filename=MIXED_DATA_PATH, collision_boost=True):
    all_data = []
    total_collisions = 0

    print(f"🚀 开始生成 {num_runs} runs 数据...")

    for run in tqdm(range(num_runs), desc="Generating"):
        # 每次都新建干净的 model（数据生成时不要加载 TCN）
        model = RetroModel(
            allow_collisions=True,
            tcn_path=None,                    # 关键：强制不加载 TCN
            width=GRID_WIDTH,
            height=GRID_HEIGHT,
            num_agents=NUM_AGENTS
        )

        run_data = []

        for step in range(steps_per_run):
            # 记录当前状态
            for agent in list(model.schedule.agents):
                agent_id = agent.unique_id
                pos = agent.pos
                rel_pos = agent.get_relative_positions()

                others_pos = [a.pos for a in model.schedule.agents if a.unique_id != agent_id]
                collision = 1 if pos in others_pos else 0

                step_data = {
                    'agent_id': agent_id,
                    'pos': pos,
                    'rel_pos': rel_pos,
                    'collision': collision,
                    'step': step
                }
                run_data.append(step_data)

                if collision and agent_id == 0:
                    total_collisions += 1

            # 执行移动
            model.step()

        all_data.append(run_data)

    # 保存
    with open(filename, 'wb') as f:
        pickle.dump(all_data, f)

    total_points = sum(len(run) for run in all_data)
    collision_rate = total_collisions / total_points if total_points > 0 else 0

    print("\n✅ 数据生成完成！")
    print(f"   Total runs: {len(all_data)}")
    print(f"   Total data points: {total_points}")
    print(f"   Total collisions (Agent 0): {total_collisions}")
    print(f"   Collision rate: {collision_rate:.4f} ({collision_rate*100:.2f}%)")

    return all_data


if __name__ == "__main__":
    generate_data(num_runs=200)