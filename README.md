<h1 align="center"> Tugas Besar 1 IF3170 Inteligensi Artifisial <br>Pencarian Solusi Diagonal Magic Cube dengan Local Search</h1>

## Table of Contents
1. [General Information](#general-information)
2. [Contributors](#contributors)
3. [Features](#features)
4. [How to Run The Program](#how-to-run-the-program)
5. [Project Structure](#project-structure)
6. [Project Status](#project-status)


## General Information
This project implements various optimization algorithms, including Simulated Annealing (SA), Genetic Algorithm (GA), and Hill Climbing Algorithms. These algorithms are applied to solve complex problems in AI, specifically focusing on finding solutions for the Diagonal Magic Cube configuration for a 5x5x5 cube. The goal is to provide a toolkit of local search algorithms to test and compare their performance in optimizing the diagonal sums and achieving a balanced configuration within the magic cube.

## Contributors
### **Kelompok 22**
|   NIM    |                  Nama                |                 Tugas                 |
| :------: | ------------------------------------ | --------------------------------------|
| 13522063 |        Shazya Audrea Taufik          | Algoritma Simulated Annealing, Laporan|
| 13522070 |          Marzuli Suhada M            | Algoritma Genetic, Laporan            |
| 13522085 |         Zahira Dina Amalia           | Algoritma Hill-Climbing, Laporan      |
| 13522108 |      Muhammad Neo Cicero Koda        | Algoritma Hill-Climbing, Laporan      |


## Features
Features that used in this program are:
| NO  | Algorithm                      | Description                                                          |
|:---:|--------------------------------|----------------------------------------------------------------------|
| 1   | Steepest Ascent Hill Climbing  | Iteratively moves to the best neighboring solution to find a local optimum in the solution space.     |
| 2   | Sideways Move Hill Climbing    | Allows moves to neighboring solutions of equal value to avoid getting stuck in local optima.     |
| 3   | Stochastic Hill Climbing       | Randomly selects a neighboring solution and moves to it if it improves the current solution.     |
| 4   | Random Restart Hill Climbing   | Repeatedly runs hill climbing from random initial states to explore different areas of the solution space.     |
| 5   | Simulated Annealing            | Uses a probabilistic approach with temperature decay to escape local optima and search globally.     |
| 6   | Genetic Algorithm              | Evolves a population of solutions using selection, crossover, and mutation to find an optimal solution.     |


## How to Run The Program
### Clone Repository
1. Open terminal
2. Clone this repository by typing `git clone https://github.com/hiirrs/Tubes1-IF3170-Kelompok22.git` in the terminal.
3. Change the directory using `cd src`.
### Run the Notebook
1. Open `ai.ipynb` using jupyter notebook or any other similar tools.
2. Click Run All to run the whole program.

## Project Structure
```bash
├── doc     
│   └── Tubes1_IF3170_Kelompok22.pdf                               
├── src 
│   ├── .ipynb_checkpoints
│   │    └── ai-checkpoint.ipynb                       
│   └── ai.ipynb                
└── README.md                       
```

## Project Status
This project has been completed and can be executed.

<br>
<h3 align="center"> THANK YOU! </h3>