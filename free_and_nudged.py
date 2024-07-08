#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 17:49:32 2024

@author: filip
"""
import numpy as np
from sklearn.datasets import load_iris
import pandas as pd
import os
import matplotlib.pyplot as plt


from eldo_support_functions import *
from non_thread_approach import *
from plots_eldo import *
from datasets import *

import argparse
import time
import subprocess
import shutil
import glob
import datetime
import random
from multiprocessing import Process

def nudged_free_phase(X, Y, input_sample, output_dir, num_iterations, input_nodes, i_sources, output_nodes, vol_sources, beta, gamma):
    ## gather the data about the network
    sample_file = input_sample

    
    node_to_inudge = dict(zip(output_nodes, i_sources))
    node_to_vdc = dict(zip(input_nodes, vol_sources))
    
    resistors_list = create_resistor_list_eldo(input_sample, save_as_new=True)
    resistor_value_dict = create_resistor_value_dict(resistors_list)
    node_voltages = create_node_voltage_dict(resistors_list)
    
    
    
    losses = None
    cond_update = None#needs to be initiliazed (also to be fixed in the future)
    sse_values = []
    all_losses = []
    all_deltaV_nudged = []
    all_deltaV_free2 = []
    all_deltaV_nudged2 = []
    accumulated_resistances = {}
    my_results_free = None
    my_results_nudged = None
    target_results = []
    
    
    
    eldo_process = start_eldo_simulation(sample_file, output_dir)
    
    for i in range(0, num_iterations):        
        X_vec = X[i, :]
        Y_vec = Y[i, :]
        
        mode = "free"
        
        
        if i == 0:
            inudge_dict = create_inudge_dict(losses, node_to_inudge, beta) #need to correct
            
            
         
        #set up everything for the free phase
        voltage_source_values = dict(zip(vol_sources, X_vec))
        #input_values = create_voltage_source_to_value_dict(node_to_vdc, X_vec)   
        set_eldo_simulation(eldo_process, mode, voltage_source_values, resistor_value_dict,  inudge_dict)
        
        
        #run simulation and wait for the results
        run_eldo_simulation(eldo_process)
        wait_for_eldos_completion(eldo_process)
        
        #extract the results
        mode = "get_voltage"
        free_node_voltages = extract_results(eldo_process, mode, node_voltages, resistor_value_dict)
        update_resistor_list(resistors_list, free_node_voltages)  # Pass these node voltages to the resistor list to connect them with appropriate nodes and keys
        voltage_matrix_free = resistor_voltage_array(resistors_list)  # Update a matrix that contains voltage differences with res (and fet) keys

        #calculate the losses and the nudging current
        losses = loss_function(Y_vec, free_node_voltages, output_nodes)  # losses is a dictionary as well
        inudge_dict = create_inudge_dict(losses, node_to_inudge, beta)  # outputs dictionary that says to what current sources what values should be applied


        # Nudge phase
        mode = "nudge"
        set_eldo_simulation(eldo_process, mode, input_values, resistor_value_dict, inudge_dict)
        run_eldo_simulation(eldo_process)
        wait_for_eldos_completion(eldo_process)


        #Extract the results
        mode = "get_voltage"
        nudge_node_voltages = extract_results(eldo_process, mode, node_voltages, resistor_value_dict)
        update_resistor_list(resistors_list, nudge_node_voltages)  # Pass these node voltages to the resistor list to connect them with appropriate nodes and keys
        voltage_matrix_nudge = resistor_voltage_array(resistors_list)  # Update a matrix that contains voltage differences with res (and fet) keys
        
        
        #update the resistances
        mode = "set_resistances"
        resistor_value_dict = extract_results(eldo_process, mode, node_voltages, resistor_value_dict)
        cond_update = calc_cond_update(voltage_matrix_free, nudge_node_voltages, gamma, beta)
        resistor_value_dict =  update_resistor_value_dict(resistor_value_dict, cond_update)
        set_eldo_simulation(eldo_process, mode, input_values, resistor_value_dict,  inudge_dict)
            
        

        

        
        sse = calculate_sse(losses, X_vec)
        sse_values.append(sse)
        all_losses.append(losses)
            
        target_results.append(Y_vec)

        #deltaVf = node_voltages['res1']
        #all_deltaV_free.append(deltaVf)
        # deltaVn = voltage_matrix_nudge[14][3]
        # all_deltaV_nudged.append(deltaVn)
        # deltaVf2 = voltage_matrix_free[12][3]
        # all_deltaV_free2.append(deltaVf2)
        # deltaVn2 = voltage_matrix_nudge[12][3]
        # all_deltaV_nudged2.append(deltaVn2)          
        
        my_results_free.append(free_node_voltages)
        my_results_nudged.append(nudge_node_voltages)  
        
        
        accumulate_resistance_values(resistor_value_dict, accumulated_resistances)
        i += 1
        
        #resistances_over_time= accumulate_resistance_values(resistance_values, resistances_over_time)
    # plot_resistance_changes(resistances_over_time)
    #selected_resistors=["res1"]
    #selected_resistors2=["res3"]
    print(f"my losses {losses}")
    #plot_combined_res_deltaV(accumulated_resistances, all_deltaV_free, all_deltaV_nudged, beta, selected_resistors)   
    #plot_combined_res_deltaV(accumulated_resistances, all_deltaV_free2, all_deltaV_nudged2, beta, selected_resistors2)   


    delete_output_directory(output_dir)
    plot_sse(sse_values, beta, gamma)
    plot_free_and_nudged(my_results, my_results_nudged, output_nodes, beta, gamma)
    plot_resistance_changes(accumulated_resistances, beta, gamma)
    plot_conductance_changes(accumulated_resistances, beta, gamma)
    plot_resistance_changes_log(accumulated_resistances, beta, gamma)
    plot_conductance_changes_log(accumulated_resistances, beta, gamma)
    #plot_deltaV_changes(all_deltaV_free, all_deltaV_nudged)
    # update_and_plot_resistances(all_iterations_updates)





def main():
    num_samples = 200
    num_iterations = 100
    X, Y = generate_dataset_2input_1output(num_samples)
    input_nodes = ["NET7", "NET9"]
    vol_sources = ["VDC1","VDC2"]
    #i_sources = ["INUDGE1", "INUDGE2"]
    #output_nodes=["NET12", "NET13"]
    i_sources = ["INUDGE"]
    output_nodes = ["NET3"]
    input_sample="/home/filip/CMOS130/simulations/sample_files/eldo_samples/6_resistors.cir"
    output_dir="/home/filip/CMOS130/simulations/simulations"
    create_output_directory(output_dir)
    beta=10e-4
    gamma=10e-4
    ## random or uniform

    nudged_free_phase(X, Y, input_sample, output_dir, num_iterations, input_nodes, i_sources, output_nodes, vol_sources, beta, gamma)    

    
if __name__ == "__main__":
    main()
