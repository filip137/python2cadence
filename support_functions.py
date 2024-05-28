import numpy as np
import os
from psf_utils import PSF
from run_spectre import *
from inputmodifier2 import *
from read_and_organise import *
from free_and_nudged import *
import matplotlib.pyplot as plt
import argparse
import time
import subprocess
import shutil
import glob
import datetime
from sklearn.datasets import load_iris
import pandas as pd

    
def move_simulation_files(source_directory, target_directory, extensions):
    # Check if the target directory exists and remove it if it does
    if os.path.exists(target_directory):
        shutil.rmtree(target_directory)
        print(f"Removed existing directory {target_directory}")

    # Create a fresh directory
    os.makedirs(target_directory, exist_ok=True)
    print(f"Created new directory {target_directory}")

    # Move files with specified extensions
    for extension in extensions:
        for file in glob.glob(os.path.join(source_directory, f'*{extension}')):
            target_file = os.path.join(target_directory, os.path.basename(file))
            shutil.move(file, target_file)
        #    print(f"Moved {file} to {target_file}")
            

def generate_dataset(num_samples, mode):
    np.random.seed(2)
    # Randomly generate currents I1 and I2 within a reasonable range
    if mode == "linear_reg":
        V1 = np.random.uniform(0, 5, num_samples)  
        V2 = np.random.uniform(0, 5, num_samples)  
    if mode == "uniform":
    # Initially create 2-dimensional arrays with half the required samples
        V1 = np.ones((num_samples // 2, 1)) * 5  
        V2 = np.ones((num_samples // 2, 1)) * 5 
    
    # Generate random data and reshape immediately to match V1 and V2's 2D shape
        rndm1 = np.random.uniform(1, 5, num_samples // 2).reshape(-1, 1)
        rndm2 = np.random.uniform(1, 5, num_samples // 2).reshape(-1, 1)
    
    # Vertically stack the original and random data
        V1 = np.vstack((V1, rndm1))
        V2 = np.vstack((V2, rndm2))

    # Flatten the arrays to make them 1-dimensional
        V1 = V1.flatten()
        V2 = V2.flatten()
    if mode == "snapshot":
        V1 = np.linspace(1, 5, num_samples)  
        V2 = np.linspace(1, 5, num_samples)        
    if mode == "zeros":
        V1 = np.random.uniform(1, 5, num_samples)  
        V2 = np.random.uniform(1, 5, num_samples)         
        VD1 = np.zeros(num_samples)  
        VD2 = np.zeros(num_samples)    
    # Calculate VD1 and VD2 based on the given formulas
    VD1 = 0.15 * V1  + 0.20 * V2 
    VD2 = 0.25 * V1  + 0.1 * V2
    # VD2=np.ones((num_samples,1))
    # Combine I1 and I2 into a single input feature matrix, and VD1 and VD2 into a targets matrix
    X = np.column_stack((V1, np.zeros([num_samples,1]), V2)) #node1 node4 node7
    Y = np.column_stack((VD1, VD2))

    return X, Y

def generate_dataset_2input_1output(num_samples, mode):
    V1 = np.zeros(num_samples)
    V2 = np.ones(num_samples)*5
    X = np.column_stack((V1, V2))
    Y = np.zeros(num_samples)
    return X, Y


def generate_dataset2(num_samples):
    # Generate inputs

    X = np.random.uniform(2, 6, num_samples)
    Y = X/5
    
    X = X.reshape(-1, 1)  # Reshape X to be num_samples x 1
    Y = Y.reshape(-1, 1)  # Reshape Y to be num_samples x 1

    return X, Y

def accumulate_resistance_values(iteration_resistances, accumulated_resistances):
    for key, value in iteration_resistances.items():
        if key in accumulated_resistances:
            accumulated_resistances[key].append(value)
        else:
            accumulated_resistances[key] = [value]
        


# Example usage:
# resistances_over_time = {'res1': [1, 2, 3], 'res2': [2, 3, 4], 'res3': [3, 4, 5]}
# all_deltaV_free = [1.0, 1.5, 2.0]
# all_deltaV_nudge = [0.8, 1.4, 1.9]
# beta = 0.5
# resistors_to_plot = ['res1', 'res3']  # Choose which resistors to plot
# plot_combined(resistances_over_time, all_deltaV_free, all_deltaV_nudge, beta, resistors_to_plot)
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
# plot_resistance_and_deltaV(resistances_over_time, all_deltaV_free, all_deltaV_nudge, beta, resistors_to_plot)

##IMPROVED RESISTANCE READER
def read_resistance_values(file_path):
    """Read resistance values from a netlist file, handling multi-line parameters with continuation,
    and store them in a dictionary.
    
    Args:
        file_path (str): The path to the file containing the netlist.

    Returns:
        dict: A dictionary with resistance names as keys and their values as float numbers.
    """
    resistances = {}
    try:
        with open(file_path, 'r') as file:
            in_parameters_block = False
            accumulated_lines = ''  # This will accumulate the parameter block lines

            for line in file:
                # Check for the start and end of the parameters block
                if 'start parameters' in line:
                    in_parameters_block = True
                    continue

                if 'end parameters' in line:
                    in_parameters_block = False
                    # Process the accumulated lines
                    # Remove continuation backslashes and strip whitespace
                    accumulated_lines = accumulated_lines.replace('\\\n', '').replace('\\', '')
                    parts = accumulated_lines.split()
                    for part in parts:
                        if part.startswith('res'):
                            key, value = part.split('=')
                            resistances[key] = float(value)
                    accumulated_lines = ''  # Reset for any further blocks
                    continue

                if in_parameters_block:
                    # Strip right side to avoid catching the backslash with trailing spaces
                    accumulated_lines += line.rstrip()  # Append line to the accumulated block

    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return resistances

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
def plot_deltaV_changes(all_deltaV_free, all_deltaV_nudge):
    """
    Plot the changes in deltaV values for 'free' and 'nudge' scenarios, and the difference of their squares.

    Args:
        all_deltaV_free (list): List of deltaV values from the free scenario.
        all_deltaV_nudge (list): List of deltaV values from the nudge scenario.
    """
    # Check if both lists are of the same length
    if len(all_deltaV_free) != len(all_deltaV_nudge):
        raise ValueError("Both lists must have the same number of elements.")

    # Calculate the difference of the squares of the values
    deltaV_diff_squares = [(x**2 - y**2) for x, y in zip(all_deltaV_free, all_deltaV_nudge)]

    # Creating the plot
    plt.figure(figsize=(10, 5))  # Set the size of the plot
    plt.plot(all_deltaV_free, label='Delta V Free', marker='o', linestyle='-')
    plt.plot(all_deltaV_nudge, label='Delta V Nudge', marker='x', linestyle='--')
    plt.plot(deltaV_diff_squares, label='Difference of Squares', marker='s', linestyle=':')

    # Adding title and labels
    plt.title('Comparison of Delta V Changes and Their Squared Differences')
    plt.xlabel('Iteration')
    plt.ylabel('Delta V and Squared Differences')

    # Adding a legend
    plt.legend()

    # Show grid
    plt.grid(True)

    # Display the plot
    plt.show()

def plot_results(target_results, my_results, output_nodes):
    num_outputs = len(output_nodes)
    
    # Ensure target_results and my_results are numpy arrays
    target_results = np.array(target_results)
    my_results = np.array(my_results)

    for i in range(num_outputs):
        plt.figure(figsize=(10, 6))
        
        # Plot target results
        plt.plot(target_results[:, i], 'o-', label=f'Target {output_nodes[i]}')
        
        # Plot current results
        plt.plot(my_results[:, i], 's-', label=f'Current {output_nodes[i]}')
        
        plt.title(f'Results for {output_nodes[i]}')
        plt.xlabel('Iteration')
        plt.ylabel('Voltage (V)')
        plt.legend()
        
        plt.tight_layout()
        plt.show()


   
def plot_free_and_nudged(my_results_free, my_results_nudged, output_nodes, beta, gamma):
    # Validate input
    if my_results_free is None or my_results_nudged is None:
        raise ValueError("Input data cannot be None")
    if len(my_results_free) != len(my_results_nudged):
        raise ValueError("Input data must have the same length")

    # Number of iterations
    n_of_iter = len(my_results_free)
    x = np.linspace(1, n_of_iter, n_of_iter)

    # Plotting data for each output node in a separate figure
    for i, node in enumerate(output_nodes):
        plt.figure(figsize=(10, 5))
        plt.plot(x, my_results_free[:, i], label=f'Free Results ({node})')
        plt.plot(x, my_results_nudged[:, i], label=f'Nudged Results ({node})')
        plt.plot(x, my_results_free[:, i] - my_results_nudged[:, i], label=f"Difference ({node})")
        
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
    
    
def plot_resistance_and_voltages(resistances_over_time, beta, selected_resistors, X):
    """
    Plot the resistance values and corresponding voltage data for the first set of iterations,
    using separate y-axes for resistance and voltage due to differing scales.

    Args:
        resistances_over_time (dict): Dictionary where keys are resistor labels and values are lists of resistance values over iterations.
        beta (float): A parameter that might affect the title or other aspects of the plot.
        selected_resistors (list): List of resistor labels to plot.
        X (np.array): 2D array where columns are voltages at different nodes (e.g., node1, node4, node7) for each measurement iteration.
    """
    fig, ax1 = plt.subplots(figsize=(12, 8))  # Set the size of the plot

    # Plot resistance changes for selected resistors on the primary y-axis
    for resistor in selected_resistors:
        if resistor in resistances_over_time:
            values = resistances_over_time[resistor]
            iterations = range(1, len(values) + 1)
            ax1.plot(iterations, values, marker='o', linestyle='-', label=f"Resistance - {resistor}")

    ax1.set_xlabel('Iteration Number')  # X-axis label
    ax1.set_ylabel('Resistance Value (Ohms)', color='tab:blue')  # Primary y-axis label
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_yscale('log')
    ax1.grid(True)

    # Create a second y-axis for the voltage data
    ax2 = ax1.twinx()  
    ax2.set_ylabel('Voltage (Volts)', color='tab:red')  # Secondary y-axis label
    ax2.plot(range(1, len(X) + 1), X[:, 0]-X[:, 2], label='Voltage difference', linestyle='--', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title(f'Resistance and Voltage Changes Over Iterations for beta={beta}')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right', title='Legend')
    plt.show()  # Display the plot

    
def plot_results_and_Y(data, num_iterations, Y, beta):
    # Ensure data and Y are NumPy arrays for consistent shape handling
    data = np.array(data)
    Y = np.array(Y[0:num_iterations])
    Y = np.atleast_2d(Y)
    if Y.shape[0] == 1:
        Y = Y.T

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
        plt.plot(iterations, data[:, i], label=f'Modeled Output {i+1}', marker='o', linestyle='-')
    
    # Plot each column of Y as a separate line
    for j in range(Y.shape[1]):
        plt.plot(iterations, Y[:, j], label=f'True Output {j+1}', marker='x', linestyle='--')

    # Adding titles and labels
    plt.title(f"Results over Iterations for beta={beta}")
    plt.xlabel('Iteration')
    plt.ylabel('Measured Values')
    plt.legend()  # This adds a legend using the labels specified in the plot commands
    
    # Show grid
    plt.grid(True)
    
    # Display the plot
    plt.show()


    
def plot_sse(sse_values, num_iterations, beta, gamma):
    """
    Plot the Sum of Squared Errors (SSE) for each iteration and display the mean of the last 5 SSE values.

    Args:
        sse_values (list): List of SSE values for each iteration.
        num_iterations (int): Total number of iterations.
        beta (float): Parameter value used to indicate experimental conditions or configuration.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_iterations + 1), sse_values, marker='o', linestyle='-', color='blue')
    plt.title(f"Sum of Squared Errors per Iteration for beta={beta} and gamma={gamma} ")
    plt.xlabel('Iteration')
    plt.ylabel('Sum of Squared Errors (SSE)')
    plt.grid(True)
    
    # Calculate the mean of the last 5 SSE values
    if len(sse_values) >= 5:
        last_5_mean = sum(sse_values[-5:]) / 5
        # Display the mean of the last 5 SSE values on the plot
        plt.axhline(y=last_5_mean, color='r', linestyle='--', label=f'Mean of last 5 SSE values: {last_5_mean:.2f}')
        plt.legend()  # Show the legend to explain the line

    plt.show()   

def calculate_sse(losses, X_vec):
    return np.sum(np.square(losses))

def create_snapshot(input_file, output_dir_, num_points, beta, output_nodes):
    X, Y = generate_dataset2(num_points)
    input_dir, output_dir = create_timestamped_dir(input_dir_, output_dir_)
    modes = ['Vdc']
    beta, losses, output_nodes, node_to_inudge = None
    for i in range (0,num_points):
        X_vec=np.round(X[i, :],2)
        Y_vec=np.round(X[i, :],2)
        new_file_path = f"{input_dir}/input{i + 1}.scs"
        modify_netlist_general(input_file, new_file_path, X_vec, Y_vec, cond_update, modes, beta, losses, output_nodes, node_to_inudge)  # Create the netlist for free phase
        output_directory = f"{output_dir}/output{i + 1}"  # set up output directory
        os.makedirs(output_directory, exist_ok=True) # Creating a new output directory for each iteration is definetly not ideal (cannot supress the results easily though)
        phase=f"free"
        run_spectre_simulation(new_file_path, output_directory, i, phase)
        current_results = read_all_results(result_file_free, output_nodes) 