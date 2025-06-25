# Dynamic Retrocausal Simulator
## Overview
This repository hosts a Jupyter Notebook implementation of a retrocausality simulation, investigating how predictive modeling 
and rule-based behavior shape multi-agent interactions within a constrained environment. 

The project simulates 30 agents navigating a 10×10 grid over 10 steps, employing a Temporal Convolutional Network (TCN) to forecast future moves and 
enforce rules to prevent collisions. The primary objective is to compare outcomes with and without rules, 
illustrating the effect of predictive adjustments on agent behavior.

## Methodology
### Simulation Environment: 
A 10×10 grid with 30 agents, built using the Mesa framework. Each agent selects from five possible moves (Up, Down, Right, Left, No move) per step, with 2000 simulation runs generating the training dataset.
### Data Preparation: 
Agent positions across 10 steps (11 total, including the initial position) are processed into 360,000 sequences, each comprising 5 timesteps and 60 features (2 for the agent’s position, 58 for relative positions of the 29 other agents).
### Model: 
A TCN with architecture [128, 128, 128] and kernel_size=7 predicts the next move based on 5 prior steps. Training is conducted on 360,000 samples with a batch size of 64.
### Rules:
#### With Rules: 
Agents leverage TCN predictions to avoid moves ("Turn Left if Occupied") and exhibit a bias toward the grid’s center (5, 5).
#### Without Rules: 
Agents move randomly, constrained only by grid boundaries and occupancy.
### Training: 
Executed over 52 epochs with early stopping, achieving a validation accuracy of 31.23% (random baseline: 20%).
