import pickle
import random
import torch
import numpy as np
from config import GRID_WIDTH, GRID_HEIGHT, NUM_AGENTS, MIXED_DATA_PATH
from src.model import RetroModel

def generate_data(num_runs=1000, steps_per_run=50, filename=MIXED_DATA_PATH):
    """
    Generate training data by running multiple simulations.
    Each run collects agent positions, relative positions, and collision flags.
    """
    all_data = []

    for run in range(num_runs):
        # Create model with collisions allowed for data generation
        model = RetroModel(width=GRID_WIDTH, height=GRID_HEIGHT, num_agents=NUM_AGENTS, allow_collisions=True)

        run_data = []

        for step in range(steps_per_run):
            # Record state before step
            for agent in model.schedule.agents:
                agent_id = agent.unique_id
                pos = agent.pos
                rel_pos = agent.get_relative_positions()
                # Check if this position collides with others
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

            # Step the model
            model.step()

        all_data.append(run_data)
        if (run + 1) % 100 == 0:
            print(f"Completed {run + 1}/{num_runs} runs")

    # Save to file
    with open(filename, 'wb') as f:
        pickle.dump(all_data, f)

    print(f"Data saved to {filename}")
    print(f"Total runs: {len(all_data)}")
    print(f"Total data points: {sum(len(run) for run in all_data)}")

    return all_data

if __name__ == "__main__":
    generate_data()