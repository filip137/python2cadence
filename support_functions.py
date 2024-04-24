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
            print(f"Moved {file} to {target_file}")
            

def generate_dataset(num_samples, mode):
    # Randomly generate currents I1 and I2 within a reasonable range
    if mode == "linear_reg":
        V1 = np.random.uniform(1, 5, num_samples)  
        V2 = np.random.uniform(1, 5, num_samples)  
    if mode == "snapshot":
        V1 = np.linspace(1, 5, num_samples)  
        V2 = np.linspace(1, 5, num_samples)        
    if mode == "zeros":
        V1 = np.random.uniform(1, 5, num_samples)  
        V2 = np.random.uniform(1, 5, num_samples)         
        VD1 = np.zeros(num_samples)  
        VD2 = np.zeros(num_samples)    
    # Calculate VD1 and VD2 based on the given formulas
    #VD1 = 0.15 * V1  + 0.20 * V2 
    #VD2 = 0.25 * V1  + 0.10 * V2
    # VD2=np.ones((num_samples,1))
    # Combine I1 and I2 into a single input feature matrix, and VD1 and VD2 into a targets matrix
    X = np.column_stack((V1, V2, np.zeros([num_samples,1])))
    Y = np.column_stack((VD1, VD2))

    return X, Y

def generate_dataset2(num_samples):
    # Randomly generate currents I1 and I2 within a reasonable range
    V1 = np.random.uniform(1, 5, num_samples)  # Current V1 range from 1 to 5 V
    V2 = 0  # Current V2 range from -1 to 1 V

    # VD2=np.ones((num_samples,1))
    # Combine I1 and I2 into a single input feature matrix, and VD1 and VD2 into a targets matrix
    X = np.random.uniform(1, 5, num_samples)
    Y = np.zeros((num_samples,1))

    return X, Y

def accumulate_resistance_values(iteration_resistances, accumulated_resistances):
    for key, value in iteration_resistances.items():
        if key in accumulated_resistances:
            accumulated_resistances[key].append(value)
        else:
            accumulated_resistances[key] = [value]
        
def plot_resistance_changes(accumulated_resistances):
    plt.figure(figsize=(10, 6))
    for resistor, values in accumulated_resistances.items():
        iterations = range(len(values))
        plt.plot(iterations, values, label=resistor, marker='o', linestyle='-')
    
    plt.title('Resistance Changes Over Iterations')
    plt.xlabel('Iteration Number')
    plt.ylabel('Resistance Value (Ohms)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
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
            
def plot_resistance_changes(resistances_over_time):
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

    plt.title('Resistance Changes Over Iterations')  # Title of the plot
    plt.xlabel('Iteration Number')  # X-axis label
    plt.ylabel('Resistance Value (Ohms)')  # Y-axis label
    plt.grid(True)  # Enable grid for better readability
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

def plot_results_and_Y(data, num_iterations, Y):
    # Ensure data and Y are NumPy arrays for consistent shape handling
    data = np.array(data)
    Y = np.array(Y[0:num_iterations])

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
    plt.title('Results over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Measured Values')
    plt.legend()  # This adds a legend using the labels specified in the plot commands
    
    # Show grid
    plt.grid(True)
    
    # Display the plot
    plt.show()

def plot_sse(sse_values, num_iterations):

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_iterations + 1), sse_values, marker='o', linestyle='-', color='blue')
    plt.title('Sum of Squared Errors per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Sum of Squared Errors (SSE)')
    plt.grid(True)
    
    
    plt.show()
    
    

def calculate_sse(losses, X_vec):
    return np.sum(np.square(losses))/np.sum(np.square(X_vec))