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
import random
from multiprocessing import Process

def nudged_free_phase(X, Y, og_input_sample, input_dir_, output_dir_, num_iterations, beta, output_nodes, gamma, init_weights):
    ## gather the data about the network
    new_sample_file_path = f"{og_input_sample}_modified"
    input_sample = modify_input_config(og_input_sample, new_sample_file_path, init_weights)

    
    input_dir, output_dir = create_timestamped_dir(input_dir_, output_dir_)
    node_to_inudge = create_node_to_inudge_map(input_sample)
    nudged_input_dir=f"{input_dir}_nudged"
    os.makedirs(nudged_input_dir, exist_ok=True)#directory where all the netlists for the nudged phase are stored 
    nudged_output_dir=f"{output_dir}_nudged"
    os.makedirs(nudged_output_dir, exist_ok=True)
    plots_output_dir=f"{output_dir}_plots"
    os.makedirs(plots_output_dir, exist_ok=True)
    
    resistors_list = create_resistor_list(input_sample, save_as_new=True)
    
    modes = ['Vdc']##for the first iteration
    losses=None
    cond_update= None#needs to be initiliazed (also to be fixed in the future)
    sse_values = []
    all_losses = []
    all_deltaV_free = []
    all_deltaV_nudged = []
    all_deltaV_free2 = []
    all_deltaV_nudged2 = []
    accumulated_resistances = {}
    my_results=None
    my_results_nudged=None
    for i in range(0, num_iterations):
        if i == 124:
            print("must debug")
        overall_start = time.time()
        # if X.ndim == 1:
        #     X_vec=np.round(X[i],2)
        # else:
        X_vec=np.round(X[i, :],2)
        # if Y.ndim == 1:
        #     Y_vec=np.round(Y[i],2)
        # else:
        Y_vec=np.round(Y[i, :],2)
        new_file_path = f"{input_dir}/input{i + 1}.scs"##where the input files will be stored
        if i==0:
            modify_netlist_general(input_sample, new_file_path, X_vec, Y_vec, cond_update, modes, beta, losses, output_nodes, node_to_inudge, gamma) # Create the netlist for free phase
        else:
            old_file_path = f"{input_dir}/input{i}.scs"
            modify_netlist_general(old_file_path, new_file_path, X_vec, Y_vec, cond_update, modes, beta, losses, output_nodes, node_to_inudge, gamma)
            
            
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
        modify_netlist_general(new_file_path, new_file_path_nudged, X_vec, Y_vec, cond_update, modes, beta, losses, output_nodes, node_to_inudge, gamma)
        phase=f"nudge"
        
        
        start_time = time.time()
        run_spectre_simulation(new_file_path_nudged, output_directory_nudged, i, phase) 
        end_time = time.time()
        print(f"Time taken for run_spectre_simulation (nudged phase) iteration {i}: {np.round(end_time - start_time,2)} seconds")
        result_file_nudged = os.path.join(output_directory_nudged, "dcOp.dc")  # Collect the results
        voltage_matrix_nudge = read_and_store_results(result_file_nudged, resistors_list)
        cond_update=calc_deltaR(voltage_matrix_free, voltage_matrix_nudge, gamma)
        ## all_iterations_updates.append(cond_update)
        modes = ['Vdc', 'deltaR']
        overall_end = np.round(time.time(),2)
        print(f"Total runtime of the script: {np.round(overall_end - overall_start,2)} seconds")
        sse = calculate_sse(losses, X_vec)
        sse_values.append(sse)
        all_losses.append(losses)
        
        current_results = read_all_results(result_file_free, output_nodes)
        current_results_nudged = read_all_results(result_file_nudged, output_nodes)
        
        deltaVf = voltage_matrix_free[14][3]
        all_deltaV_free.append(deltaVf)
        deltaVn = voltage_matrix_nudge[14][3]
        all_deltaV_nudged.append(deltaVn)
        deltaVf2 = voltage_matrix_free[12][3]
        all_deltaV_free2.append(deltaVf2)
        deltaVn2 = voltage_matrix_nudge[12][3]
        all_deltaV_nudged2.append(deltaVn2)          
        if my_results is None:
            my_results = current_results
        else:
            my_results = np.vstack((my_results, current_results))
            
        if my_results_nudged is None:
            my_results_nudged = current_results_nudged
        else:
            my_results_nudged = np.vstack((my_results_nudged, current_results_nudged))
          
        resistance_values=read_resistance_values(new_file_path)
        accumulate_resistance_values(resistance_values, accumulated_resistances)
        #resistances_over_time= accumulate_resistance_values(resistance_values, resistances_over_time)
    # plot_resistance_changes(resistances_over_time)
    selected_resistors=["res1"]
    selected_resistors2=["res3"]
    print(f"my losses {losses}")
    plot_combined_res_deltaV(accumulated_resistances, all_deltaV_free, all_deltaV_nudged, beta, selected_resistors)   
    plot_combined_res_deltaV(accumulated_resistances, all_deltaV_free2, all_deltaV_nudged2, beta, selected_resistors2)   

    plot_sse(sse_values, num_iterations, beta, gamma)
    plot_free_and_nudged(my_results, my_results_nudged, output_nodes, beta)
    plot_resistance_changes(accumulated_resistances, beta, gamma)
    plot_resistance_changes_log(accumulated_resistances, beta, gamma)
    plot_deltaV_changes(all_deltaV_free, all_deltaV_nudged)
    # update_and_plot_resistances(all_iterations_updates)
    log_directory = os.path.join(output_dir, "log_files")
    move_simulation_files(os.getcwd(), log_directory, ['.log', '.ahdlSimDB'])#need to be in the directory where python files are for this to work
                                                                                

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
    mode = "linear_reg"
    X, Y = generate_dataset(300, mode)
    output_nodes=["node2", "node5"]
    input_sample="/home/filip/CMOS130/simulations/sample_files/1_transistor_upenn"
    input_dir="/home/filip/CMOS130/simulations/various_tests/upenn"
    output_dir="/home/filip/CMOS130/simulations/various_tests/upenn"
    beta=0.0001
    gamma_a=[5e-10]
    ## random or uniform
    init_weights=[]
    for i in range(1, 17): 
        #resv = random.randint(60,140)
        resv = 10000
        init_weights.append({'resistor': f'res{i}', 'init_weight': resv})
    
    
    for gamma in gamma_a:
        nudged_free_phase(X, Y, input_sample,input_dir , output_dir, 150 , beta, output_nodes, gamma, init_weights)
    

    
if __name__ == "__main__":
    main()

