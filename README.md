# water-treatment-adsorption-modeling
ODE-based adsorption modeling batch reactor for Langmuir &amp; Freundlich isotherms, with synthetic dataset generation

This repository contains a student-built modeling framework for adsorption-based water treatment. The idea is to simulate a batch-reactor adsorption process using well-known isotherm models, generate synthetic datasets, and explore how different parameters affect adsorption behavior. It uses Langmuir and Freundlich isotherm models, ODE-based simulation of batch reactor adsorption over time, synthetic dataset generation for testing, and visualization tools. The goal is to provide an easy, flexible environment to experiment with adsorption without needing expensive or proprietary software.

#Why I made it

As part of my water-quality research at my community college, I wanted to learn how to use python more effectively in the lab and also needed a way to quickly simulate adsorption-based clean-up under different conditions. I couldn’t find a small, open, easy-to-use tool that did exactly what I needed, so I built this. Developing this project helped me understand adsorption isotherms, practice ODE-based modeling in Python, write reusable code, and generate synthetic data to anticipate real experiments.

#What’s included in the repo
isotherm_functions.py – Langmuir and Freundlich isotherm functions
simulation.py – ODE-based batch reactor adsorption simulation
random_generator.py – Synthetic dataset generation
fit_isotherm_functions_linear_regression.py – Scripts to fit isotherm parameters from data
eda_dashboard_graphs.py – Generates plots for exploratory data analysis
Simulation_notebook.ipynb – Jupyter notebook tying everything together
.gitignore and LICENSE – Standard project files

#How to use it
-Clone the repository
-Set up a Python environment with dependencies like NumPy, SciPy, Matplotlib, and pandas
-Open Simulation_notebook.ipynb to:
-Adjust initial conditions
-Run adsorption simulations
-Generate synthetic datasets
-Fit isotherm parameters and visualize results
-Modify the code to add new models, experiment with kinetics, or extend simulations

# What this project is not
This is not a production-ready water treatment tool. It’s meant for learning and experimentation, as well as a learning project for me. It does not account for all real-world complexities like column flow, temperature variations, or multi-component adsorption. Parameter values are often idealized or synthetic.

#Possible extensions
Add more isotherm models like BET or Temkin
Model continuous reactors or fixed-bed columns
Include mass-transfer limitations or multi-component adsorption
Link simulations to real experimental data for validation
Build a simple GUI for easier parameter control and visualization
