# Dynamic Retrocausal Simulator

## Overview
This repository implements a retrocausality simulation exploring how predictive modeling and rule-based behavior affect multi-agent interactions in a constrained grid environment.

The project simulates agents navigating a grid, using a Temporal Convolutional Network (TCN) to predict and avoid collisions. The goal is to demonstrate how retrocausal reasoning (predicting future states to influence current behavior) can lead to emergent coordination without explicit communication.

## Project Structure
```
Dynamic-Retrocausal-Simulator/
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
├── main.py                   # Main script for complete pipeline
├── src/                      # Source code
│   ├── agents.py            # Agent behavior and rules
│   ├── model.py             # Simulation model and TCN architecture
│   ├── tcn.py               # TCN training code
│   ├── data_gen.py          # Data generation utilities
│   ├── evaluate.py          # Model evaluation and analysis
│   └── visualize.py         # Visualization and animation
├── data/                    # Data files
├── models/                  # Trained model files
├── results/                 # Results and outputs
└── notebooks/               # Jupyter notebooks for exploration
    └── Explore_and_Debug.ipynb
```

## Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation
```bash
pip install -r requirements.txt
```

### Run Complete Pipeline
```bash
python main.py
```
This will:
1. Generate training data
2. Train the TCN model
3. Evaluate performance
4. Create visualization

### Individual Steps
- Generate data: `python -m src.data_gen`
- Train model: `python -m src.tcn`
- Evaluate: `python -m src.evaluate`
- Visualize: `python -m src.visualize`

## Methodology

### Simulation Environment
- Configurable grid size (default: 5×5)
- Variable number of agents (default: 3)
- Built using Mesa framework

### Data Preparation
- Agents collect position and relative position data over multiple timesteps
- Sequences of 5 timesteps used for prediction
- Features: agent position (2) + relative positions of other agents (10)

### Model
- Temporal Convolutional Network (TCN)
- Predicts collision probability based on recent history
- Used for retrocausal decision making

### Rules
- **With Rules**: Agents use TCN predictions to avoid collisions
- **Without Rules**: Random movement with basic collision avoidance

## Results
Check the `results/` directory for:
- Model evaluation metrics
- Confusion matrices
- Feature importance analysis
- Simulation animations (GIF)

## Development
Use the Jupyter notebook in `notebooks/` for interactive exploration and debugging.
