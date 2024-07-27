#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import datetime
"""
Created on Tue Jul 23 18:52:01 2024

@author: filip
"""

def main_function(sample_file, parameters_dict):
    create_cir(sample_file_path)
    write_parameters(parameters_dict, sample_file)
    

def create_cir(output_file_path):
    """
    Creates a .cir file with a header that includes the current date and time.

    :param output_file_path: Path where the new .cir file should be saved.
    """
    # Get current date and time
    current_datetime = datetime.datetime.now().strftime("%b %d %H:%M:%S %Y")

    # Define the header with the current date and time
    input_header = f"""***
*** Generated for: eldoD
*** Generated on: {current_datetime}
*** Design library name: tests
*** Design cell name: kendal_non_linear_moons_easy
*** Design view name: schematic
.GLOBAL
"""
def write_param_header(sample_file):
    output_param_header = """.LIB /cao/DK/ST/HCMOS9A_10.9/Addon_NVM_H9A@2018.4.1/tools/eldo/model_oxram/OxRRAM.lib OxRRAM_TT
.LIB /home/filip/CMOS130/corners.eldo 
.LIB /home/filip/Documents/MyDiode.lib """
    with open(sample_file,"w") as file:
        file.write(output_param_header)
def write_neuron(sample_file):
    neuron_definition = """.SUBCKT NEURON VIN VOUT
    D0 VIN NET6 diode1
    D1 NET7 VIN diode1
    V2 NET6 0 DC VDIODE1
    V3 NET7 0 DC VDIODE2
    F0 0 VIN EVCVS1 {1/AMP}
    EVCVS1 VOUT 0 VIN 0 AMP
.ENDS"""
    with open(sample_file,"w") as file:
        file.write(neuron_definition)
def write_parameters(parameters_dict, sample_file):
    
    with open(sample_file,"w") as file:
    
        for key, value in parameters_dict.items():
            line = f".PARAM {key}=1"
            file.write(line)