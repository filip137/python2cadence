#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import time
import subprocess
import shutil
import glob
import datetime
from sklearn.datasets import load_iris
import pandas as pd
"""
PLOTTING FUNCTIONS 

"""

def accumulate_resistance_values(iteration_resistances, accumulated_resistances):
    for key, value in iteration_resistances.items():
        if key in accumulated_resistances:
            accumulated_resistances[key].append(value)
        else:
            accumulated_resistances[key] = [value]
        
def plot_combined_res_deltaV(resistances_over_time, all_deltaV_free, all_deltaV_nudge, beta, resistors_to_plot):
    """
    Plot the resistance values for specified resistors across iterations and the difference of the squares of
    deltaV values for 'free' and 'nudge' scenarios, each on its own y-axis.

    Args:
        resistances_over_time (dict): Dictionary where keys are resistor labels (e.g., 'res1', 'res2', etc.)
                                      and values are lists of resistance values over iterations.
        all_deltaV_free (list): List of deltaV values from the free scenario.
        all_deltaV_nudge (list): List of deltaV values from the nudge scenario.
        beta (float): A parameter for annotation in the plot.
        resistors_to_plot (list): List of resistor labels to be plotted.
    """
    if len(all_deltaV_free) != len(all_deltaV_nudge):
        raise ValueError("Both lists must have the same number of elements.")

    # Calculate the difference of the squares of the values
    deltaV_diff_squares = [(x**2 - y**2) for x, y in zip(all_deltaV_nudge, all_deltaV_free)]

    # Create a figure and axis object
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plotting resistance changes for selected resistors on the primary y-axis
    for resistor in resistors_to_plot:
        if resistor in resistances_over_time:
            values = resistances_over_time[resistor]
            iterations = range(1, len(values) + 1)
            ax1.plot(iterations, values, marker='o', linestyle='-', label=f"{resistor} Resistance")
        else:
            print(f"Warning: {resistor} not found in resistance data.")
    ax1.set_title(f'Resistance and Delta V Changes for beta={beta}')
    ax1.set_xlabel('Iteration Number')
    ax1.set_ylabel('Resistance Value (Ohms)', color='tab:blue')
    ax1.set_yscale('log')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True)
    ax1.legend(loc='upper left')

    # Create a second y-axis for the difference of squares
    ax2 = ax1.twinx()
    ax2.plot(iterations, deltaV_diff_squares, 'r-', marker='s', label='Difference of Squares')
    ax2.set_ylabel('Difference of Squares', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.legend(loc='upper right')

    # Show the plot
    plt.tight_layout()
    plt.show()

# Example usage:
# resistances_over_time = {'res1': [1, 2, 3], 'res2': [2, 3, 4]}
# all_deltaV_free = [1.0, 1.5, 2.0]
# all_deltaV_nudge = [0.8, 1.4, 1.9]
# beta = 0.5
# resistors_to_plot = ['res1', 'res2']  # Choose which resistors to plot
# _and_deltaV(resistances_over_time, all_deltaV_free, all_deltaV_nudge, beta, resistors_to_plot)


def accumulate_resistance_values(current_resistances, resistances_over_time):
    """
    Accumulates resistance values from a single iteration into a cumulative dictionary.

    Args:
        current_resistances (dict): Dictionary containing resistance values for the current iteration.
        resistances_over_time (dict): Dictionary where keys are resistor labels and values are lists of resistance values across iterations.
    """
    for key, value in current_resistances.items():
        if key in resistances_over_time:
            resistances_over_time[key].append(value)
        else:
            resistances_over_time[key] = [value]
 
            
def plot_conductance_changes(resistances_over_time, beta, gamma):
    """
    Plot the conductance values across iterations as converted from the resistances_over_time dictionary.

    Args:
        resistances_over_time (dict): Dictionary where keys are resistor labels (e.g., 'res1', 'res2', etc.)
                                      and values are lists of resistance values over iterations.
        beta (float): Parameter value used to indicate experimental conditions or configuration.
        gamma (float): Parameter value used to indicate experimental conditions or configuration.
    """
    conductances_over_time = {}
    for resistor, resistances in resistances_over_time.items():
        conductances_over_time[resistor] = [1 / r if r != 0 else np.inf for r in resistances]

    plt.figure(figsize=(12, 8))  # Set the size of the plot

    # Generate a plot for each resistor in the dictionary
    for resistor, values in conductances_over_time.items():
        # Create an x-axis range based on the number of iterations
        iterations = range(1, len(values) + 1)
        # Plot the conductance changes over iterations
        plt.plot(iterations, values, marker='o', linestyle='-', label=resistor)

    plt.title(f'Conductance Changes Over Iterations for beta={beta} and gamma={gamma}')  # Title of the plot
    plt.xlabel('Iteration Number')  # X-axis label
    plt.ylabel('Conductance Value (Siemens)')  # Y-axis label
    plt.grid(True)  # Enable grid for better readability
    plt.legend(title='Resistor')  # Add a legend with a title
    plt.show()  # Display the plot

def plot_conductance_changes_log(resistances_over_time, beta, gamma):
    """
    Plot the conductance values across iterations as converted from the resistances_over_time dictionary,
    using a logarithmic scale for the y-axis.

    Args:
        resistances_over_time (dict): Dictionary where keys are resistor labels (e.g., 'res1', 'res2', etc.)
                                      and values are lists of resistance values over iterations.
        beta (float): Parameter value used to indicate experimental conditions or configuration.
        gamma (float): Parameter value used to indicate experimental conditions or configuration.
    """
    conductances_over_time = {}
    for resistor, resistances in resistances_over_time.items():
        conductances_over_time[resistor] = [1 / r if r != 0 else np.inf for r in resistances]

    plt.figure(figsize=(12, 8))  # Set the size of the plot

    # Generate a plot for each resistor in the dictionary
    for resistor, values in conductances_over_time.items():
        # Create an x-axis range based on the number of iterations
        iterations = range(1, len(values) + 1)
        # Plot the conductance changes over iterations
        plt.plot(iterations, values, marker='o', linestyle='-', label=resistor)
    
def plot_free_and_nudged(my_results_free, my_results_nudged, output_nodes, beta, gamma):
    # Validate input
    if not my_results_free or not my_results_nudged:
        raise ValueError("Input data cannot be None or empty")
    if len(my_results_free) != len(my_results_nudged):
        raise ValueError("Input data must have the same length")
    
    # Number of iterations
    n_of_iter = len(my_results_free)
    x = np.linspace(1, n_of_iter, n_of_iter)

    # Extract data for each output node
    for node in output_nodes:
        free_results = np.array([result[node] for result in my_results_free])
        nudged_results = np.array([result[node] for result in my_results_nudged])

        plt.figure(figsize=(10, 5))
        plt.plot(x, free_results, label=f'Free Results ({node})')
        plt.plot(x, nudged_results, label=f'Nudged Results ({node})')
        plt.plot(x, free_results - nudged_results, label=f"Difference ({node})")
        
        # Adding titles and labels
        plt.title(f"Results over Iterations for {node} (beta={beta}, gamma={gamma})")
        plt.xlabel('Iteration')
        plt.ylabel('Measured Value at output')

        # Legend
        plt.legend(loc='best')  # Improved legend placement

        # Grid
        plt.grid(True)

        # Display the plot
        plt.show()

    
def plot_resistance_changes(resistances_over_time, beta, gamma):
    """
    Plot the resistance values across iterations as stored in the resistances_over_time dictionary.

    Args:
        resistances_over_time (dict): Dictionary where keys are resistor labels (e.g., 'res1', 'res2', etc.)
                                      and values are lists of resistance values over iterations.
    """
    plt.figure(figsize=(12, 8))  # Set the size of the plot

    # Generate a plot for each resistor in the dictionary
    for resistor, values in resistances_over_time.items():
        # Create an x-axis range based on the number of iterations
        iterations = range(1, len(values) + 1)
        # Plot the resistance changes over iterations
        plt.plot(iterations, values, marker='o', linestyle='-', label=resistor)

    plt.title(f'Resistance Changes Over Iterations for beta={beta} and gamma={gamma}')  # Title of the plot
    plt.xlabel('Iteration Number')  # X-axis label
    plt.ylabel('Resistance Value (Ohms)')  # Y-axis label
    plt.grid(True)  # Enable grid for better readability
    plt.legend(title='Resistor')  # Add a legend with a title
    plt.show()  # Display the plot




def plot_resistance_changes_log(resistances_over_time, beta, gamma):
    """
    Plot the resistance values across iterations as stored in the resistances_over_time dictionary,
    using a logarithmic scale for the y-axis.

    Args:
        resistances_over_time (dict): Dictionary where keys are resistor labels (e.g., 'res1', 'res2', etc.)
                                      and values are lists of resistance values over iterations.
        beta (float): Parameter value used to indicate experimental conditions or configuration.
    """
    plt.figure(figsize=(12, 8))  # Set the size of the plot

    # Generate a plot for each resistor in the dictionary
    for resistor, values in resistances_over_time.items():
        # Create an x-axis range based on the number of iterations
        iterations = range(1, len(values) + 1)
        # Plot the resistance changes over iterations
        plt.plot(iterations, values, marker='o', linestyle='-', label=resistor)

    plt.title(f'Resistance Changes Over Iterations for beta={beta} and gamma={gamma}')  # Title of the plot
    plt.xlabel('Iteration Number')  # X-axis label
    plt.ylabel('Resistance Value (Ohms)')  # Y-axis label
    plt.yscale('log')  # Set the y-axis to logarithmic scale
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Enable grid for better readability, compatible with log scale
    plt.legend(title='Resistor')  # Add a legend with a title
    plt.show()  # Display the plot


def plot_sse_values(sse_values, gamma, beta):
    """
    Plot the SSE values over iterations with gamma and beta values in the title.

    Parameters:
    sse_values (list): A list of SSE values.
    gamma (float): The gamma value.
    beta (float): The beta value.

    Returns:
    None
    """
    iterations = list(range(1, len(sse_values) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, sse_values, marker='o', linestyle='-', color='b', label='Root of Squared Losses')
    plt.xlabel('Iteration')
    plt.ylabel('Root of Squared Losses')
    plt.title(f'Root of Squared Losses Over Iterations (Gamma: {gamma}, Beta: {beta})')
    plt.legend()
    plt.grid(True)
    plt.show()



def plot_results(data, num_iterations, Y):
    # Ensure data is a NumPy array for consistent shape handling
    data = np.array(data)

    if data.ndim == 1:
        data = data.reshape(-1, 1)  # Convert 1D array to 2D array with one column if necessary
    
    if data.shape[0] != num_iterations:
        print("Warning: Number of iterations does not match the number of rows in the data.")
    
    # Generate an array representing the number of iterations
    iterations = np.arange(num_iterations)
    
    # Plotting
    plt.figure(figsize=(10, 5))
    
    # Plot each column in the data as a separate line
    for i in range(data.shape[1]):
        plt.plot(iterations, data[:, i], label=f'Output {i+1}', marker='o', linestyle='-')
    
    # Adding titles and labels
    plt.title('Results over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Measured Values')
    plt.legend()  # This adds a legend using the labels specified in the plot commands
    
    # Show grid
    plt.grid(True)
    
    # Display the plot
    plt.show()
