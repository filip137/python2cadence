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
from build_cis_file import neural_network



import argparse
import time
import subprocess
import shutil
import glob
import datetime
import random
from sklearn.model_selection import train_test_split
from multiprocessing import Process

def nudged_free_phase(X, Y, input_sample, num_of_epochs, beta, gamma, debug, scale_factor, boundary):
    
    
    input_nodes = ["VIN1", "VIN2", "VIN3", "VIN4", "VIN5", "VIN6", "VIN7", "VIN8"]   # Example input nodes
    output_nodes = ["V_Y1", "V_Y2"]

    vol_sources = ["VDC1", "VDC2", "VDC3", "VDC4", "VDC5", "VDC6", "VDC7", "VDC8"]   # Example voltage sources
    i_sources = ["INUDGE_Y1", "INUDGE_Y2"]

    output_dir="/home/filip/simulations/simulations"

    
    
    
    ## gather the data about the network
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.5, random_state=42, shuffle = False)  # 60% training, 40% for validation and test
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42, shuffle = False)# Splits remaining 40% into 20% validation, 20% test
    #X_test, Y_test = generate_all_combinations_xor()
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
    batch_size = 10
    for j in range(0, num_of_epochs):
        
        for i in range(0, X_train.shape[0], batch_size):
        #for i in range(0, X_train.shape[0]):   
            X_batch = X_train[i:i+batch_size]
            Y_batch = Y_train[i:i+batch_size]
            
            batch_losses = []
            total_loss = {}
            
            for k in range(0, X_batch.shape[0]):
                mode = "free"
                X_vec = X_batch[k, :]
                Y_vec = Y_batch[k, :] 
               # print(f"k: {k}, X_batch[k]: {X_batch[k]}")
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
            #losses = loss_function_xor(Y_vec, free_node_voltages, output_nodes) # losses is a dictionary as well
                batch_loss = loss_function_moon(Y_vec, free_node_voltages, output_nodes, boundary)
                for key, value in batch_loss.items():
                    if key in total_loss:
                        total_loss[key] += value
                    else:
                        total_loss[key] = value
                
                
            losses = {key: value / batch_size for key, value in total_loss.items()}
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
            cond_update = calc_cond_update(voltage_matrix_free, voltage_matrix_nudge, gamma, beta) #be careful, the variables need to have a certain name
            resistor_value_dict =  update_resistor_value_dict(resistor_value_dict, cond_update) 
            mode = "set_resistances"
            set_resistances(eldo_process, resistor_value_dict, debug)
            losses_list = list(losses.values())
            nudge_node_voltages_list = list(nudge_node_voltages.values())
            if debug:
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
        
        disable_current_sources(eldo_process, inudge_dict, debug)
        accuracy = validate(eldo_process, X_test, Y_test, vol_sources, debug, resistor_value_dict, node_to_vdc, node_voltages, output_nodes, boundary)
        accuracy_after_epoch.append(accuracy)
        draw_grid(eldo_process, X_test, Y_test, vol_sources, debug, resistor_value_dict, node_to_vdc, node_voltages, output_nodes, j,scale_factor, boundary)
        
        
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

def run_free_phase(eldo_process, X_train, Y_train, vol_sources, node_to_inudge, beta, debug):
    sse_values, losses_records, results_free = [], [], []
    for i in range(X_train.shape[0]):
        input_values = dict(zip(vol_sources, X_train[i]))
        disable_current_sources(eldo_process, create_inudge_dict({}), debug)
        set_input_voltages(eldo_process, input_values, debug)
        run_eldo_simulation(eldo_process, debug)
        wait_for_eldos_completion(eldo_process, debug)
        free_node_voltages = extract_results(eldo_process, "get_voltage", {}, {}, debug)
        losses = loss_function_xor(Y_train[i], free_node_voltages, node_to_inudge.keys())
        sse_values.append(calculate_single_sse(losses))
        losses_records.append(losses)
        results_free.append(free_node_voltages)
    return sse_values, losses_records, results_free



def draw_grid(process, X_val, Y_val, vol_sources, debug, resistor_value_dict, node_to_vdc, node_voltages, output_nodes, j,scale_factor, boundary):
    
    # Define bounds of the domain
    min1, max1 = X_val[:, 0].min() - 0.1, X_val[:, 0].max() + 0.1
    min2, max2 = X_val[:, 1].min() - 0.1, X_val[:, 1].max() + 0.1
    
    num_points = 30

    x1grid = np.linspace(min1, max1, num_points)
    x2grid = np.linspace(min2, max2, num_points)

# Create a meshgrid from the grid points
    xx, yy = np.meshgrid(x1grid, x2grid)
    grid = np.c_[xx.ravel(), yy.ravel()]

# Make predictions for the grid
    y_predictions = predict(process, grid, vol_sources, debug, resistor_value_dict, node_to_vdc, node_voltages, output_nodes, boundary)

# Convert predictions into an array and reshape back into a grid
    y_predictions = np.array(y_predictions)
    zz = y_predictions.reshape(xx.shape)

# Plot the grid of x, y, and z values as a surface
    contour = plt.contourf(xx, yy, zz, cmap='Paired')
    cbar = plt.colorbar(contour)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Class 0', 'Class 1'])
    Y_val = Y_val.ravel()
# Separate the points by class
    class0 = X_val[Y_val == 0]
    class1 = X_val[Y_val == 1]

# Scatter plot for the validation set with legend
    plt.scatter(class0[:, 0], class0[:, 1], c='blue', edgecolor='k', marker='o', s=20, label='Class 0')
    plt.scatter(class1[:, 0], class1[:, 1], c='red', edgecolor='k', marker='o', s=20, label='Class 1')

# Add legend
    plt.legend()

# Add titles and labels
    plt.title(f"Decision Boundary with True Samples after epoch {j} and scale factor{scale_factor}")
    plt.xlabel('VDC1')
    plt.ylabel('VDC2')

# Show plot
    plt.show()
    
    
    
   

def predict(process, X_val, vol_sources, debug, resistor_value_dict, node_to_vdc, node_voltages, output_nodes, boundary):
    
    set_resistances(process, resistor_value_dict, debug)
    
    pred_outputs = []
    true_outputs = []
    vol_values = []
    
    for i in range(0, X_val.shape[0]):        
        X_vec = X_val[i, :]
        

        
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
      
        #calculate the losses and the nudging current
        #losses = loss_function_moon(Y_vec, free_node_voltages, output_nodes) # losses is a dictionary as well
        predicted_output = predicted_value(free_node_voltages, output_nodes, boundary)
        pred_outputs.append(predicted_output)
        vol_values.append(voltage_values(free_node_voltages, output_nodes))

    return pred_outputs



def validate(process, X_val, Y_val, vol_sources, debug, resistor_value_dict, node_to_vdc, node_voltages, output_nodes, boundary):
    
    set_resistances(process, resistor_value_dict, debug)
    
    pred_outputs = []
    true_outputs = []
    vol_values = []
    
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
        losses = loss_function_moon(Y_vec, free_node_voltages, output_nodes, boundary) # losses is a dictionary as well
        predicted_output = predicted_value(free_node_voltages, output_nodes, boundary)
        pred_outputs.append(predicted_output)
        true_outputs.append(Y_vec)
        vol_values.append(voltage_values(free_node_voltages, output_nodes))
    average_voltage_values = list(zip(vol_values, Y_val))
    accuracy = calculate_accuracy(pred_outputs, true_outputs)
    #accuracy_after_epoch.append(accuracy)
    print(f"Accuracy at the end of the epoch {accuracy}")
    return accuracy

def generate_all_combinations_xor():
    """
    Generate all possible combinations for the XOR function with specific encoding:
    0 is encoded as -4 and 1 as 4.

    Returns:
    X (numpy.ndarray): The encoded input pairs.
    y (numpy.ndarray): The corresponding XOR outputs.
    """
    # Define the four possible combinations for a binary input
    combinations = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    
    # Apply the encoding: 0 -> -4 and 1 -> 4
    X_encoded = np.where(combinations == 0, -2, 2)
    
    # Compute the XOR output
    # XOR is true if the bits are different
    y = np.logical_xor(combinations[:, 0], combinations[:, 1]).astype(int)
    
    # Apply encoding to the output as well: 0 -> -4, 1 -> 4
    y_encoded = y.reshape(-1, 1)
    
    return X_encoded, y_encoded   


def main():
    num_samples = 1000
    num_of_epochs = 2
    scale_factor = 2
    X, Y = prepare_moons_data(num_samples, noise=0.1, random_state=41)
    #X = 2*X
    X, Y = generate_biased_inputs(X, Y, scale_factor)
    #X, Y = generate_biased_pos_neg_inputs(X, Y, scale_factor)
    #X, Y = generate_pos_neg_inputs(X, Y, scale_factor)
    #X, Y = generate_xor_data(num_samples)
    
    
    input_nodes = ["VIN1", "VIN2", "VIN3", "VIN4", "VIN5", "VIN6", "VIN7", "VIN8"]   # Example input nodes
    output_nodes = ["V_Y1", "V_Y2"]

    vol_sources = ["VDC1", "VDC2", "VDC3", "VDC4", "VDC5", "VDC6", "VDC7", "VDC8"]   # Example voltage sources
    i_sources = ["INUDGE_Y1", "INUDGE_Y2"]

    amp = 1
    camp = 1
    
    v_diode_pos_values = np.linspace(0.5, 5, 5)  # Example range and number of values
    v_diode_neg_values = -v_diode_pos_values
    v_diode_list = list(zip(v_diode_pos_values, v_diode_neg_values))
    
    
    
    
    

    #i_sources = ["INUDGE1"]
    #output_nodes = ["NET07"]
    #input_sample ="/home/filip/simulations/sample_files/eldo_samples/kendal_non_linear.cir"
    #input_sample = "/home/filip/CMOS130/simulations/kendal_non_linear_moons/eldoD/schematic/netlist/kendal_non_linear_moons.cir"
    output_dir="/home/filip/simulations/simulations"
    create_output_directory(output_dir)
    beta = 1e-5
    gamma = 1e-8
    boundary = 0.5
    #beta_list = [5*beta1, 6*beta1, 7*beta1, 8*beta1, 9*beta1, 10*beta1]

    gamma_list = np.linspace(0.1,5,10)*gamma
    scale_factor_list =np.linspace(0.5,10,15)*scale_factor
    boundary_list = 0.5
    debug = False
    input_sample = "/home/filip/simulations/sample_files/eldo_samples/virtuoso netlists tests/kendall_moons_cadence.cir"
    ## random or uniform
    for v_diode_pos, v_diode_neg in v_diode_list:
       # nn = neural_network(input_nodes, vol_sources, i_sources, v_diode_pos, v_diode_neg, amp, camp)
       # nn.write_to_file(input_sample)
        nudged_free_phase(X, Y, input_sample, num_of_epochs, beta, gamma, debug, scale_factor, boundary)    
    #nudged_free_phase(X, Y, input_sample, output_dir, num_of_epochs, input_nodes, i_sources, output_nodes, vol_sources, beta1, gamma1, debug)
    
if __name__ == "__main__":
    main()
