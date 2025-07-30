# STARIM: Straight-Through and Autoregressive Influence Estimation for Influence Maximization

This repository contains the official implementation for the paper: **"Straight-Through and Autoregressive Influence Estimation for Influence Maximization"**.

## Overview

STARIM proposes a novel model-based optimization framework for influence maximization (IM). STARIM leverages Graph Neural Network(GNN) and an autoregressive approach to model the continuous evolution of node states, and incorporates Straight-Through Estimation(STE) to enable efficient end-to-end optimization over a continuous seed space. This design allows for the inclusion of additional constraints, such as varying costs for each seed node, or targeted IM, where influence is maximized over a predefined target set.

## Getting Started

To get started with the code and reproduce our results, please follow the instructions below.

```bash
pip install -r requirements.txt
```

You can test the performance of the two main variants of STARIM: STAR-M (Model-based) and STAR-N (Neural-based).

# 1. Test STAR-M (Model-based STARIM)
STAR-M directly leverages a mechanistic propagation model for influence estimation, making it straightforward to run.

```bash
python STAR-M.py
```

# 2. Test STAR-N (Neural-based STARIM)
STAR-N utilizes a learned neural network to model the propagation process. This requires generating specific training data (propagation trajectories) and training the neural propagation model beforehand.

Step 1: Generate propagation trajectory training data .This script will generate the necessary data files in the 'data' directory.

```bash
cd data

python generate_train_data_traj.py

cd ..
```

Step 2: Train the neural propagation model. This step trains the GNN-based propagation model.

```bash
python train_model.py -pwd /path/to/your/STARIM -propagation_data ic
```

Step 3: Run STAR-N for influence maximization. After the model is trained, execute STAR-N to find the optimal seed set.

```bash
python STAR-N.py -pwd /path/to/your/STARIM -propagation_data ic
```