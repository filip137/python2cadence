#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 17:49:32 2024

@author: filip
"""
import numpy as np
import os
from psf_utils import PSF
from run_spectre import *
from inputmodifier2 import *
from read_and_organise import *
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
            
def generate_dataset(num_samples):
    # Randomly generate currents I1 and I2 within a reasonable range
    V1 = np.random.uniform(1, 5, num_samples)  # Current V1 range from -1 to 1 V
    V2 = np.random.uniform(1, 5, num_samples)  # Current V2 range from -1 to 1 V

    # Calculate VD1 and VD2 based on the given formulas
    VD1 = 0.15 * V1  + 0.20 * V2 
    VD2 = 0.25 * V1 * 1 + 0.10 * V2
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

def nudged_free_phase(X, Y, input_sample, input_dir_, output_dir_, num_iterations, beta, output_nodes):
    ## gather the data about the network
    # Generate random X (20 columns, 2 rows) and Y (20 elements)
    # Preprocess and read the file content
    input_dir, output_dir = create_timestamped_dir(input_dir_, output_dir_)
    resistors_list = create_resistor_list(input_sample, save_as_new=True)
    node_to_inudge = create_node_to_inudge_map(input_sample)
    nudged_input_dir=f"{input_dir}_nudged"
    os.makedirs(nudged_input_dir, exist_ok=True)#directory where all the netlists for the nudged phase are stored 
    nudged_output_dir=f"{output_dir}_nudged"
    os.makedirs(nudged_output_dir, exist_ok=True)#same for the nudged phase
    modes = ['Vdc']##for the first iteration
    losses=1
    cond_update= 1#needs to be initiliazed (also to be fixed in the future)
    sse_values = []
    all_losses = []
    all_iterations_updates = []
    accumulated_resistances = {}
    my_results=None
    for i in range(0, num_iterations):
        overall_start = time.time()
        X_vec=np.round(X[i],2)
        Y_vec=np.round(Y[i],2)
        new_file_path = f"{input_dir}/input{i + 1}.scs"##where the input files will be stored
        if i==0:
            modify_netlist_general(input_sample, new_file_path, X_vec, Y_vec, cond_update, modes, beta, losses, output_nodes, node_to_inudge)  # Create the netlist for free phase
        else:
            old_file_path = f"{input_dir}/input{i}.scs"
            modify_netlist_general(old_file_path, new_file_path, X_vec, Y_vec, cond_update, modes, beta, losses, output_nodes, node_to_inudge) 
        output_directory = f"{output_dir}/output{i + 1}"  # set up output directory
        os.makedirs(output_directory, exist_ok=True) # Creating a new output directory for each iteration is definetly not ideal (cannot supress the results easily though)
        phase=f"free"
        start_time = time.time()
        run_spectre_simulation(new_file_path, output_directory, i, phase)  # Run the free phase
        end_time = time.time()
        print(f"Time taken for run_spectre_simulation (free phase) iteration {i}: {np.round(end_time - start_time,2)} seconds")
        result_file_free = os.path.join(output_directory, "dcOp.dc")  # Collect the results
        voltage_matrix_free = read_and_store_results(result_file_free, resistors_list)
        modes=['Inudge']#this is passed to the function which modifies the netlist
        losses=loss_function(result_file_free, Y_vec, output_nodes)#
        new_file_path_nudged=f"{nudged_input_dir}/input_nudged{i + 1}.scs"
        output_directory_nudged = f"{nudged_output_dir}/output{i + 1}"
        modify_netlist_general(new_file_path, new_file_path_nudged, X_vec, Y_vec, cond_update, modes, beta, losses, output_nodes, node_to_inudge)
        phase=f"nudge"
        start_time = time.time()
        run_spectre_simulation(new_file_path_nudged, output_directory_nudged, i, phase) 
        end_time = time.time()
        print(f"Time taken for run_spectre_simulation (nudged phase) iteration {i}: {np.round(end_time - start_time,2)} seconds")
        result_file_nudged = os.path.join(output_directory_nudged, "dcOp.dc")  # Collect the results
        voltage_matrix_nudge = read_and_store_results(result_file_nudged, resistors_list)
        cond_update=calc_deltaR(voltage_matrix_free, voltage_matrix_nudge, beta)
        all_iterations_updates.append(cond_update)
        modes = ['Vdc', 'deltaR']
        overall_end = np.round(time.time(),2)
        print(f"Total runtime of the script: {np.round(overall_end - overall_start,2)} seconds")
        sse = calculate_sse(losses, X_vec)
        sse_values.append(sse)
        all_losses.append(losses)
        current_results = read_all_results(result_file_free, output_nodes)    
        if my_results is None:
            my_results = current_results
        else:
            my_results = np.vstack((my_results, current_results))
        resistance_values=read_resistance_values(new_file_path)
        accumulate_resistance_values(resistance_values, accumulated_resistances)
        #resistances_over_time= accumulate_resistance_values(resistance_values, resistances_over_time)
    # plot_resistance_changes(resistances_over_time)
    plot_results(my_results, num_iterations)
    plot_sse(sse_values, num_iterations)
    plot_resistance_changes(accumulated_resistances)
    # update_and_plot_resistances(all_iterations_updates)
    log_directory = os.path.join(output_dir, "log_files")
    move_simulation_files(os.getcwd(), log_directory, ['.log', '.ahdlSimDB'])#need to be in the directory where python files are for this to work
    
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

# def main():
#       X = np.random.rand(20, 2) #need to move this in the inputs
#       Y = np.random.rand(20) #need to move this in the inputs
#       parser = argparse.ArgumentParser(description="Run Free Phase Simulation")
#       parser.add_argument('input_sample', type=str, help="Path to the input sample netlist file")
#       parser.add_argument('input_dir', type=str, help="Directory where input files will be stored")
#       parser.add_argument('output_dir', type=str, help="Directory where output directories will be created")
#       parser.add_argument('num_iterations', type=int, help="Number of iterations to run")
#       parser.add_argument('--beta', type=float, default=0.1, help="Beta value for simulation ")

#       args = parser.parse_args()

#       nudged_free_phase(X, Y, args.input_sample, args.input_dir, args.output_dir, args.num_iterations, args.beta)



##IMPROVED RESISTANCE READER
# def read_resistance_values(file_path):
#     """Read resistance values from a netlist file, handling multi-line parameters with continuation,
#     and store them in a dictionary.
    
#     Args:
#         file_path (str): The path to the file containing the netlist.

#     Returns:
#         dict: A dictionary with resistance names as keys and their values as float numbers.
#     """
#     resistances = {}
#     try:
#         with open(file_path, 'r') as file:
#             in_parameters_block = False
#             accumulated_lines = ''  # This will accumulate the parameter block lines

#             for line in file:
#                 # Check for the start and end of the parameters block
#                 if '//start parameters' in line:
#                     in_parameters_block = True
#                     continue

#                 if '//end parameters' in line:
#                     in_parameters_block = False
#                     # Process the accumulated lines
#                     # Remove continuation backslashes and strip whitespace
#                     accumulated_lines = accumulated_lines.replace('\\\n', '').replace('\\', '')
#                     parts = accumulated_lines.split()
#                     for part in parts:
#                         if part.startswith('res'):
#                             key, value = part.split('=')
#                             resistances[key] = float(value)
#                     accumulated_lines = ''  # Reset for any further blocks
#                     continue

#                 if in_parameters_block:
#                     # Strip right side to avoid catching the backslash with trailing spaces
#                     accumulated_lines += line.rstrip()  # Append line to the accumulated block

#     except FileNotFoundError:
#         print(f"Error: The file {file_path} does not exist.")
#     except Exception as e:
#         print(f"An error occurred: {e}")

#     return resistances

def read_resistance_values(file_path):
    """Read resistance values from a netlist file and store them in a dictionary.
    
    Args:
        file_path (str): The path to the file containing the netlist.

    Returns:
        dict: A dictionary with resistance names as keys and their values as float numbers.
    """
    resistances = {}
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if line.strip().startswith('parameters'):
                    parts = line.split()
                    for part in parts:
                        if part.startswith('res'):
                            key, value = part.split('=')
                            resistances[key] = float(value)
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


# def update_and_plot_resistances(iterations_updates):
#     """
#     Updates resistances over multiple iterations and plots the results, accepting structured numpy arrays as input.
    
#     Parameters:
#     - iterations_updates: list of numpy structured arrays, where each array contains updates
#       for each iteration with dtype=[('resistor', '<U10'), ('cond_update', '<f8')].
#     """
#     # Initial setup
#     initial_resistance = 500
#     resistors = [f"res{i}" for i in range(1, 17)]
#     resistance_history = {res: [initial_resistance] for res in resistors}
    
#     # Process each set of updates per iteration
#     for updates in iterations_updates:
#         # Update resistance values
#         for update in updates:
#             resistor, change = update['resistor'], update['cond_update']
#             # Add new resistance value to history
#             current_value = resistance_history[resistor][-1] + change
#             resistance_history[resistor].append(current_value)
    
#     # Plotting results
#     plt.figure(figsize=(15, 10))
#     for res, values in resistance_history.items():
#         plt.plot(values, label=res)
#     plt.xlabel('Iteration')
#     plt.ylabel('Resistance (Ohms)')
#     plt.title('Resistance Changes Over Iterations')
#     plt.legend(title="Resistors", bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

def plot_results(data, num_iterations):
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
   
def create_timestamped_dir(input_dir, output_dir):
    # Get the current date and time
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    # Append the timestamp to the directory paths
    timestamped_input_dir = f"{input_dir}_{timestamp}"
    timestamped_output_dir = f"{output_dir}_{timestamp}"

    # Optional: Create these directories if they do not exist
    os.makedirs(timestamped_input_dir, exist_ok=True)
    os.makedirs(timestamped_output_dir, exist_ok=True)

    return timestamped_input_dir, timestamped_output_dir


def main():
    X, Y = generate_dataset2(150)
    output_nodes=["net1"]
    input_sample="/home/filip/CMOS130/simulations/2_resistors/spectre/schematic/netlist/input.scs"
    input_dir="/home/filip/CMOS130/simulations/various_tests/2_resistances"
    output_dir="/home/filip/CMOS130/simulations/various_tests/2_resistances_outputs"
    beta=0.1
    nudged_free_phase(X, Y, input_sample,input_dir , output_dir, 20 , beta,output_nodes)

    

    
if __name__ == "__main__":
    main()

