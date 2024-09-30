# Monte Carlo Tree Search (MCTS)

## Overview

This repository contains an implementation of **Monte Carlo Tree Search (MCTS)**. The main codes can be found in the `coding` folder, with examples of problem instances located in the `Instances` folder.

## Folder Structure

### 1. Instances
In this folder, you will find two sample problem instances:
- `1.in`
- `2.in`

For more instances, you can download additional files using the link provided in the `Instances.txt` file.

### 2. Coding
The main code for running the MCTS algorithm consists of three key files:
- **Main.py**: This is where you can adjust the MCTS parameters before running the code.
- **MCTS.py**: Contains the implementation of the MCTS algorithm.
- **Node.py**: Implements the node structure used within the MCTS algorithm.

### 3. Parallelization Versions
If you wish to run the MCTS with parallelization, use the following scripts:
- **Main_Parallelisation.py**
- **Main_Parralelisation.py**

These scripts can be executed in the same manner as `Main.py`, and they will generate log files containing the steps taken by the MCTS process.

## How to Run

1. Adjust the parameters of the MCTS algorithm in `Main.py` as needed.
2. Run the code using the following command:
   ```bash
   python Main.py
