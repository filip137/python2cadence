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
from support_functions import *
import matplotlib.pyplot as plt
import argparse
import time
import subprocess
import shutil
import glob
import datetime



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
    accumulated_resistances = {}
    my_results=None
    for i in range(0, num_iterations):
        overall_start = time.time()
        X_vec=np.round(X[i, :],2)
        Y_vec=np.round(Y[i, :],2)
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
        ## all_iterations_updates.append(cond_update)
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
    plot_results_and_Y(my_results, num_iterations, Y)
    plot_sse(sse_values, num_iterations)
    plot_resistance_changes(accumulated_resistances)
    # update_and_plot_resistances(all_iterations_updates)
    log_directory = os.path.join(output_dir, "log_files")
    move_simulation_files(os.getcwd(), log_directory, ['.log', '.ahdlSimDB'])#need to be in the directory where python files are for this to work
    


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





# def read_resistance_values(file_path):
#     """Read resistance values from a netlist file and store them in a dictionary.
    
#     Args:
#         file_path (str): The path to the file containing the netlist.

#     Returns:
#         dict: A dictionary with resistance names as keys and their values as float numbers.
#     """
#     resistances = {}
#     try:
#         with open(file_path, 'r') as file:
#             for line in file:
#                 if line.strip().startswith('parameters'):
#                     parts = line.split()
#                     for part in parts:
#                         if part.startswith('res'):
#                             key, value = part.split('=')
#                             resistances[key] = float(value)
#     except FileNotFoundError:
#         print(f"Error: The file {file_path} does not exist.")
#     except Exception as e:
#         print(f"An error occurred: {e}")
    
#     return resistances






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

def create_snapshot(input_file, output_dir_, num_points, beta, output_nodes):
    X, Y = generate_dataset(num_points)
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
    X, Y = generate_dataset(150, "zeros")
    output_nodes=["node2", "node5"]
    input_sample="/home/filip/CMOS130/simulations/upenn/spectre/schematic/netlist/input.scs"
    input_dir="/home/filip/CMOS130/simulations/various_tests/upenn_2nd"
    output_dir="/home/filip/CMOS130/simulations/various_tests/upenn_2nd"
    beta_a=[0.7]
    for beta in beta_a:
        nudged_free_phase(X, Y, input_sample,input_dir , output_dir, 50 , beta, output_nodes)

    

    
if __name__ == "__main__":
    main()

