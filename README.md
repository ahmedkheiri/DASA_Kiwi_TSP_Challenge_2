# Monte Carlo Tree Search (MCTS)

## Overview

This repository contains an implementation of a **Monte Carlo Tree Search (MCTS)** to solve the TSP Challenge 2.0 proposed by Kiwi.com in 2018. The main codes can be found in the `Coding` folder, with examples of problem instances located in the `Instances` folder.

Please cite the following paper if you used any of these resources: Arnaud Da Silva and Ahmed Kheiri (in press) A Monte Carlo tree search for the optimisation of flight connections. DASA'24

## Folder Structure

### 1. Instances

In the folder [Instances](./Instances), you will find the first 8 instances discussed in the paper:

- `1.in`
- `2.in`
- `3.in`
- `4.in`
- `5.in`
- `6.in`
- `7.in`
- `8.in`

The problem has a total of 14 instances. You can download the remaining instances from the following link:
[Download Additional Instances](https://drive.google.com/file/d/1NV0LvmUByFR2MLlp7Z9EtEhF_KyLSCxS/view)

### 2. Coding

The main code for running the MCTS algorithm consists of three key files located in the `Coding` folder:

- **Main.py**: This file allows you to adjust the MCTS parameters before running the algorithm.
- **MCTS.py**: Contains the core implementation of the MCTS algorithm.
- **Node.py**: Implements the node structure used within the MCTS algorithm.

#### Parallelisation Versions

If you wish to run the MCTS algorithm with parallelisation, use the following script instead of **Main.py**:

- **Main_Parallelisation.py**

These scripts can be executed in the same manner as `Main.py`, and they will generate log files containing the steps taken by the MCTS process.

#### How to Run

1. Adjust the parameters of the MCTS algorithm in `Main.py` as needed.
2. Run the MCTS file.

### 3. Key Findings

In this project, we applied Monte Carlo Tree Search (MCTS) to tackle the Kiwi.com Travelling Salesman Problem 2.0. The main highlights from the computational results are as follows:

- **Instance Results**: The MCTS algorithm produced solutions that matched or improved upon the state-of-the-art solutions for several problem instances.
  - For instance $I_8$, a new best solution was found, surpassing the previous best by 0.52%.
  - For instances $I_1$, $I_2$, and $I_3$, the algorithm matched the best-known solutions.
  - For instance $I_7$, the solution found was within 3.19% of the state-of-the-art result.

#### Best Results vs State of the Art

| Instance | Best Known | Best Found | Gap (%) | Mean  | Std |
| -------- | ---------- | ---------- | ------- | ----- | --- |
| $I_1$    | 1396       | **1396**   | 0       | 1396  | 0   |
| $I_2$    | 1498       | **1498**   | 0       | 1498  | 0   |
| $I_3$    | 7672       | **7672**   | 0       | 7672  | 0   |
| $I_4$    | 13952      | 15361      | 10.1    | 15361 | 0   |
| $I_5$    | 690        | -          | -       | -     | -   |
| $I_6$    | 2159       | -          | -       | -     | -   |
| $I_7$    | 30937      | 31924      | 3.19    | 30937 | 0   |
| $I_8$    | 4052       | **4037**   | -0.52   | 4052  | 0   |

- **Selection Policies**:
  - The **UCB1-Tuned** selection policy generally outperformed the classic **UCB** policy by balancing exploration and exploitation more effectively. However, UCB1-Tuned took longer to converge compared to UCB.
- **Expansion Policies**:

  - The **top-k** expansion policy proved most effective for finding solutions, especially in larger instances like $I_7$ and $I_8$.
  - Maintaining a balanced **expansion ratio** of 0.5 worked well for more complex instances.

- **Simulation Policies**:

  - The **greedy simulation policy** consistently performed best across various instances, providing the fastest convergence to near-optimal solutions.
  - The **random simulation policy** was effective for smaller instances but less so for larger, more complex problems.

- **Parallelisation**: Parallelisation using multiple cores significantly improved the performance of the MCTS algorithm, particularly for larger instances. Simulations performed in parallel provided more accurate estimates of node values, resulting in faster convergence.
