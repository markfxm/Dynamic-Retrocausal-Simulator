import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import torch
import numpy as np
from simulation_core import RetroModel

def visualize_comparison(steps=20, output_file='simulation_comparison.gif'):
    # Initialize models
    # Case 1: Random (No Rule, collisions allowed)
    model_random = RetroModel(allow_collisions=True, width=5, height=5, num_agents=3)
    
    # Case 2: Retrocausal (TCN Rule, collisions allowed but avoided)
    model_retro = RetroModel(allow_collisions=True, tcn_path='MixedData/tcn_model.pt', width=5, height=5, num_agents=3)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    def update(frame):
        # Step models
        model_random.step()
        model_retro.step()
        
        # Clear axes
        ax1.clear()
        ax2.clear()
        
        # Setup Grid 1 (Random)
        ax1.set_xlim(-0.5, 4.5)
        ax1.set_ylim(-0.5, 4.5)
        ax1.set_xticks(range(5))
        ax1.set_yticks(range(5))
        ax1.grid(True)
        ax1.set_title(f"Random Agents (Step {frame})\nCollisions: {model_random.agent0_collisions if hasattr(model_random, 'agent0_collisions') else 'N/A'}")
        
        # Setup Grid 2 (Retrocausal)
        ax2.set_xlim(-0.5, 4.5)
        ax2.set_ylim(-0.5, 4.5)
        ax2.set_xticks(range(5))
        ax2.set_yticks(range(5))
        ax2.grid(True)
        ax2.set_title(f"Retrocausal Agents (Step {frame})\nCollisions: {model_retro.agent0_collisions if hasattr(model_retro, 'agent0_collisions') else 'N/A'}") # simulation_core classes might miss collision tracking
        
        # Helper to plot agents
        def plot_agents(model, ax):
            # Check collisions for display
            pos_count = {}
            for agent in model.schedule.agents:
                pos = agent.pos
                pos_count[pos] = pos_count.get(pos, 0) + 1
            
            for agent in model.schedule.agents:
                x, y = agent.pos
                color = 'blue' if agent.unique_id == 0 else 'green'
                if pos_count[(x, y)] > 1:
                    color = 'red' # Collision!
                
                ax.plot(x, y, 'o', markersize=20, color=color, alpha=0.7)
                ax.text(x, y, str(agent.unique_id), color='white', ha='center', va='center', fontweight='bold')
                
        plot_agents(model_random, ax1)
        plot_agents(model_retro, ax2)

    print(f"Generating animation ({steps} steps)...")
    ani = animation.FuncAnimation(fig, update, frames=steps, repeat=False)
    
    # Save
    try:
        ani.save(output_file, writer='pillow', fps=2)
        print(f"Animation saved to {output_file}")
    except Exception as e:
        print(f"Error saving animation: {e}")

if __name__ == "__main__":
    # Ensure tracking of collisions is in RetroModel if not already
    # simulation_core.py RetroModel init didn't explicitly initialize collision counters or step logic for counting.
    # Let's monkey patch or rely on visual inspection if counting isn't there.
    # Actually, let's just rely on visual red dots.
    visualize_comparison()
