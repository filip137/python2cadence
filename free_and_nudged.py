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
from inputmodifier2 import modify_netlist_general
from read_and_organise import *
import argparse
import time
import subprocess
import shutil
import glob

def move_simulation_files(source_directory, target_directory, extensions):
    os.makedirs(target_directory, exist_ok=True)
    for extension in extensions:
        for file in glob.glob(os.path.join(source_directory, f'*{extension}')):
            shutil.move(file, target_directory)
            print(f"Moved {file} to {target_directory}")

def nudged_free_phase(X, Y, input_sample, input_dir, output_dir, num_iterations, beta, input_nodes, output_nodes):
    ## gather the data about the network
    # Generate random X (20 columns, 2 rows) and Y (20 elements)
    resistors_list = create_resistor_list(input_sample, save_as_new=True)
    node_to_inudge = create_node_to_inudge_map(input_sample)
    os.makedirs(input_dir, exist_ok=True)#directory where all the netlists for free phase are stored 
    nudged_input_dir=f"{input_dir}_nudged"
    os.makedirs(nudged_input_dir, exist_ok=True)#directory where all the netlists for the nudged phase are stored 
    os.makedirs(output_dir, exist_ok=True)#directory with the directories where the free phase results are stored. each iteration creates one (can't limit the output easily)
    nudged_output_dir=f"{output_dir}_nudged"
    os.makedirs(nudged_output_dir, exist_ok=True)#same for the nudged phase
    modes = ['Vdc']##for the first iteration
    losses=1
    cond_update= 1#needs to be initiliazed (also to be fixed in the future)
    for i in range(0, num_iterations):
        overall_start = time.time()
        X_vec=np.round(X[i,:],2)
        Y_vec=np.round(Y[i],2)
        new_file_path = f"{input_dir}/input{i + 1}.scs"  ##where the input files will be stored
        modify_netlist_general(input_sample, new_file_path, X_vec, Y_vec, cond_update, modes, beta, losses, outputs, node_to_inudge)  # Create the netlist for free phase
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
        modify_netlist_general(input_sample, new_file_path_nudged, X_vec, Y_vec, cond_update, modes, beta, losses, outputs, node_to_inudge)
        phase=f"nudge"
        start_time = time.time()
        run_spectre_simulation(new_file_path_nudged, output_directory_nudged, i, phase) 
        end_time = time.time()
        print(f"Time taken for run_spectre_simulation (nudged phase) iteration {i}: {np.round(end_time - start_time,2)} seconds")
        result_file_nudged = os.path.join(output_directory_nudged, "dcOp.dc")  # Collect the results
        voltage_matrix_nudge = read_and_store_results(result_file_nudged, resistors_list)
        cond_update=calc_deltaR(voltage_matrix_free, voltage_matrix_nudge, beta)
        modes = ['Vdc', 'deltaR']
        overall_end = np.round(time.time(),2)
        print(f"Total runtime of the script: {np.round(overall_end - overall_start,2)} seconds")
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

        
def main():
    X = np.random.rand(20, 3) #need to move this in the inputs
    Y = np.random.rand(20, 2) #need to move this in the inputs
    input_sample="/home/filip/CMOS130/simulations/upenn/spectre/schematic/netlist/input.scs"
    input_dir="/home/filip/CMOS130/simulations/various_tests/_nudge_input_files"
    output_dir="/home/filip/CMOS130/simulations/various_tests/nudge_output_files"
    nudged_free_phase(X, Y, input_sample,input_dir , output_dir, 5, 0.1)

    

    
if __name__ == "__main__":
    main()

