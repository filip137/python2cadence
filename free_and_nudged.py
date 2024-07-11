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
from plots_eldo import *
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
from sklearn.model_selection import train_test_split
from multiprocessing import Process

def nudged_free_phase(X, Y, input_sample, output_dir, num_of_epochs, input_nodes, i_sources, output_nodes, vol_sources, beta, gamma, debug):
    ## gather the data about the network
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.4, random_state=42)  # 60% training, 40% for validation and test
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)  # Splits remaining 40% into 20% validation, 20% test
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
    my_results_free = []
    my_results_nudged = []
    target_results = []
    
    
    
    eldo_process = start_eldo_simulation(sample_file, output_dir, debug=False)
    #initialize the resistances
    mode = "set_resistances"
    low_bound = 1e1
    up_bound = 1e4
    initialize_res(resistor_value_dict, low_bound, up_bound, uni_res = None, mode1 = "random")
    set_eldo_initial(eldo_process, mode, resistor_value_dict, debug)
    accuracy_after_epoch = []
    for j in range(0, num_of_epochs):
        
        for i in range(0, X_train.shape[0]):        
            X_vec = X_train[i, :]
            Y_vec = Y_train[i, :]
        
            mode = "free"
        
        
            if i == 0:
                inudge_dict = create_inudge_dict(losses, node_to_inudge, beta) #need to correct
                mode = "get_resistance"
                resistor_value_dict = extract_results(eldo_process, mode, node_voltages, resistor_value_dict, debug)
                accumulate_resistance_values(resistor_value_dict, accumulated_resistances)
                mode = "free"
                
            
         
        #set up everything for the free phase
            voltage_source_values = dict(zip(vol_sources, X_vec))
            #input_values = create_voltage_source_to_value_dict(node_to_vdc, X_vec)  
            input_values = dict(zip(vol_sources, X_vec))
            disable_current_sources(eldo_process, inudge_dict, debug)
            set_input_voltages(eldo_process, input_values, debug)
        
        
        #run simulation and wait for the results
            run_eldo_simulation(eldo_process, debug)
            wait_for_eldos_completion(eldo_process, debug)
        
        #extract the results
            mode = "get_voltage"
            free_node_voltages = extract_results(eldo_process, mode, node_voltages, resistor_value_dict, debug) #contains voltages at nodes at the end of free phase
            update_resistor_list(resistors_list, free_node_voltages)  # Pass these node voltages to the resistor list to connect them with appropriate nodes and keys
            voltage_matrix_free = resistor_voltage_array(resistors_list)  # Update a matrix that contains voltage differences with res (and fet) keys

        #calculate the losses and the nudging current
            #losses = loss_function(Y_vec, free_node_voltages, output_nodes)
            losses = loss_function_xor(Y_vec, free_node_voltages, output_nodes) # losses is a dictionary as well
        
        #inj_curr = 10e-7
        #inudge_dict = create_inudge_dict_const(losses, node_to_inudge, beta, inj_curr)  # outputs dictionary that says to what current sources what values should be applied
            inudge_dict = create_inudge_dict(losses, node_to_inudge, beta)

        # Nudge phase
            mode = "nudge"
            set_currents_nudge_mode(eldo_process, inudge_dict, debug)
            run_eldo_simulation(eldo_process, debug)
            wait_for_eldos_completion(eldo_process, debug)


        #Extract the results
            mode = "get_voltage"
            nudge_node_voltages = extract_results(eldo_process, mode, node_voltages, resistor_value_dict, debug)#contains voltages at the nodes at the end of nudge phase
            update_resistor_list(resistors_list, nudge_node_voltages)  # Pass these node voltages to the resistor list to connect them with appropriate nodes and keys
            voltage_matrix_nudge = resistor_voltage_array(resistors_list)  # Update a matrix that contains voltage differences with res (and fet) keys
        
        
        #update the resistances
            mode = "get_resistance"
            resistor_value_dict = extract_results(eldo_process, mode, node_voltages, resistor_value_dict, debug)
            cond_update = calc_cond_update(voltage_matrix_free, voltage_matrix_nudge, gamma, beta)
            resistor_value_dict =  update_resistor_value_dict(resistor_value_dict, cond_update) 
            mode = "set_resistances"
            set_resistances(eldo_process, resistor_value_dict, debug)
            losses_list = list(losses.values())
            nudge_node_voltages_list = list(nudge_node_voltages.values())
            if any(abs(voltage) > 1e2 for voltage in nudge_node_voltages_list):
                print("error")
        
        #test on validation data
        
        
            sse = calculate_single_sse(losses)
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
        
        accuracy = validate(eldo_process,  X_val, Y_val, vol_sources, debug, resistor_value_dict, node_to_vdc, node_voltages, output_nodes)
        accuracy_after_epoch.append(accuracy)
        
        
    # plot_resistance_changes(resistances_over_time)
    #selected_resistors=["res1"]
    #selected_resistors2=["res3"]
    print(f"my losses {losses}")
    #plot_combined_res_deltaV(accumulated_resistances, all_deltaV_free, all_deltaV_nudged, beta, selected_resistors)   
    #plot_combined_res_deltaV(accumulated_resistances, all_deltaV_free2, all_deltaV_nudged2, beta, selected_resistors2)   

    quit_eldo_simulation(eldo_process, debug)
    delete_output_directory(output_dir)
    plot_sse_values(sse_values, gamma, beta)
    plot_free_and_nudged(my_results_free, my_results_nudged, output_nodes, beta, gamma)
    plot_resistance_changes(accumulated_resistances, beta, gamma)
    plot_conductance_changes(accumulated_resistances, beta, gamma)
    plot_resistance_changes_log(accumulated_resistances, beta, gamma)
    plot_conductance_changes_log(accumulated_resistances, beta, gamma)
    plot_accuracy(accuracy_after_epoch)
    #plot_deltaV_changes(all_deltaV_free, all_deltaV_nudged)
    # update_and_plot_resistances(all_iterations_updates)


def validate(process, X_val, Y_val, vol_sources, debug, resistor_value_dict, node_to_vdc, node_voltages, output_nodes):
    
    set_resistances(process, resistor_value_dict, debug)
    
    pred_outputs = []
    true_outputs = []
    
    
    for i in range(0, X_val.shape[0]):        
        X_vec = X_val[i, :]
        Y_vec = Y_val[i, :]
    

        
        #set up everything for the free phase
        voltage_source_values = dict(zip(vol_sources, X_vec))
        #input_values = create_voltage_source_to_value_dict(node_to_vdc, X_vec)   
        
        input_values = dict(zip(vol_sources, X_vec))
        set_input_voltages(process, input_values, debug)
        
        
        #run simulation and wait for the results
        run_eldo_simulation(process, debug)
        wait_for_eldos_completion(process, debug)
        
        #extract the results
        mode = "get_voltage"
        free_node_voltages = extract_results(process, mode, node_voltages, resistor_value_dict, debug) #contains voltages at nodes at the end of free phase
        #update_resistor_list(resistors_list, free_node_voltages)  # Pass these node voltages to the resistor list to connect them with appropriate nodes and keys
        #voltage_matrix_free = resistor_voltage_array(resistors_list)  # Update a matrix that contains voltage differences with res (and fet) keys

        #calculate the losses and the nudging current
        losses = loss_function_xor(Y_vec, free_node_voltages, output_nodes) # losses is a dictionary as well
        predicted_output = predicted_value(free_node_voltages, output_nodes)
        pred_outputs.append(predicted_output)
        true_outputs.append(Y_vec)

    accuracy = calculate_accuracy(pred_outputs, true_outputs)
    #accuracy_after_epoch.append(accuracy)
    print(f"Accuracy at the end of the epoch {accuracy}")
    return accuracy
    
def main():
    num_samples = 400
    num_of_epochs = 10
    X, Y = generate_xor_data(num_samples)
    #X, Y = generate_xor_data(num_samples)
    input_nodes = ["NET6", "NET1"]
    vol_sources = ["VDC1","VDC2"]
    i_sources = ["INUDGE1", "INUDGE2"]
    output_nodes = ["NET07", "NET08"]
    #i_sources = ["INUDGE1"]
    #output_nodes = ["NET07"]
    input_sample="/home/filip/simulations/sample_files/eldo_samples/kendal_non_linear.cir"
    output_dir="/home/filip/simulations/simulations"
    create_output_directory(output_dir)
    beta1 = 10e-6
    gamma1 = 1e-6
    #beta_list = [5*beta1, 6*beta1, 7*beta1, 8*beta1, 9*beta1, 10*beta1]

    gamma_list = np.linspace(0.1,1.5,10)*gamma1
    debug = False
    ## random or uniform

    for gamma in gamma_list:
        nudged_free_phase(X, Y, input_sample, output_dir, num_of_epochs, input_nodes, i_sources, output_nodes, vol_sources, beta1, gamma, debug)    
    #nudged_free_phase(X, Y, input_sample, output_dir, num_of_epochs, input_nodes, i_sources, output_nodes, vol_sources, beta1, gamma1, debug)
    
if __name__ == "__main__":
    main()
