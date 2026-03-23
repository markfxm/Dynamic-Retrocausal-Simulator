#!/usr/bin/env python3
"""
Main script for Dynamic Retrocausal Simulator
Complete pipeline: Generate Data → Train Model → Evaluate → Visualize
"""

import os
import sys
import torch
from config import *
from src.data_gen import generate_data
from src.tcn import train
from src.evaluate import evaluate_model
from src.visualize import create_animation

def main():
    print("=== Dynamic Retrocausal Simulator ===")
    print("Starting complete pipeline...")

    # Step 1: Generate training data
    print("\n1. Generating training data...")
    if not os.path.exists(MIXED_DATA_PATH):
        generate_data(num_runs=1000, steps_per_run=50, filename=MIXED_DATA_PATH)
    else:
        print(f"Data file {MIXED_DATA_PATH} already exists, skipping generation.")

    # Step 2: Train the TCN model
    print("\n2. Training TCN model...")
    if not os.path.exists(TCN_MODEL_PATH):
        train()
    else:
        print(f"Model file {TCN_MODEL_PATH} already exists, skipping training.")

    # Step 3: Evaluate the model
    print("\n3. Evaluating model performance...")
    evaluate_model()

    # Step 4: Create visualization
    print("\n4. Creating visualization...")
    create_animation()

    print("\n=== Pipeline completed successfully! ===")
    print("Check the results/ directory for outputs.")

if __name__ == "__main__":
    main()