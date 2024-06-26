import numpy as np
from read_and_organise import *


##SIMPLE ONE LINE VERSION
# def modify_netlist_general(original_file_path, new_file_path, X_vec, Y_vec, cond_update, modes, beta, losses, outputs, node_to_inudge):
#     # Ensure modes is a list to simplify processing
#     if not isinstance(modes, list):
#         modes = [modes]

#     # Convert X_vec to a list if it is a scalar
#     if not isinstance(X_vec, list):
#         X_vec = [X_vec] * len(outputs)  # Assume the length of outputs matches the needed length of X_vec

#     # Read the original netlist file
#     with open(original_file_path, 'r') as file:
#         lines = file.readlines()

#     # Process and modify lines
#     modified_lines = []
    
#     for line in lines:
#         # Process only lines that start with "parameters"
#         if line.strip().startswith("parameters"):
#             parts = line.split()

#             # Update Vdc values if requested
#             if 'Vdc' in modes:
#                 for i, vdc_value in enumerate(X_vec, start=1):
#                     parts = [f'Vdc{i}={vdc_value}' if part.startswith(f'Vdc{i}=') else part for part in parts]

#             # Update Inudge value if requested
#             if 'Inudge' in modes:
#                 parts = update_inudge_values(parts, node_to_inudge, outputs, losses, beta)

#             # Update deltaR values if requested
#             if 'deltaR' in modes:
#                 for update in cond_update:
#                     res_name = update['resistor']  # Get the resistor name
#                     delta = update['cond_update']  # Get the corresponding delta value
#                     for i, part in enumerate(parts):
#                         if part.startswith(res_name + '='):
#                             original_value = float(part.split('=')[1])
#                             new_value = original_value - (1/original_value**2) * delta * 10
#                             parts[i] = f'{res_name}={new_value}'
#                             break  # Found and updated resistor, move to next

#             # Replace the line with updated parts
#             line = ' '.join(parts) + '\n'

#         # Append the processed or unprocessed line to modified_lines
#         modified_lines.append(line)

#     # Write the modified content to a new file
#     with open(new_file_path, 'w') as file:
#         file.writelines(modified_lines)

#     print(f'Modified netlist saved to {new_file_path}')

def modify_input_config(sample_file_path, new_sample_file_path, init_weights):
    # Read the original netlist file
    with open(sample_file_path, 'r') as file:
        lines = file.readlines()

    modified_lines = []
    in_parameters_block = False
    block_lines = []

    for line in lines:
        if '//start parameters' in line:
            in_parameters_block = True
            block_lines.append(line)
            continue

        if '//end parameters' in line:
            in_parameters_block = False
            block_lines.append(line)
            # Process the collected lines in the block
            for block_line in block_lines:
                parts = block_line.split()
                for res in init_weights:
                    res_name = res['resistor']  # Get the resistor name
                    r0 = res['init_weight']  # Get the corresponding initial weight value
                    for i, part in enumerate(parts):
                        if part.startswith(res_name + '='):
                            parts[i] = f'{res_name}={r0}'
                            break
                modified_lines.append(' '.join(parts) + '\n')
            block_lines = []
            continue

        if in_parameters_block:
            block_lines.append(line)
        else:
            modified_lines.append(line)

    # Write the modified content to a new file
    with open(new_sample_file_path, 'w') as file:
        file.writelines(modified_lines)

    print(f'Modified sample file saved to {new_sample_file_path}')
    
    return new_sample_file_path

def modify_netlist_general(original_file_path, new_file_path, X_vec, Y_vec, cond_update, modes, beta, losses, outputs, node_to_inudge, gamma):
    # Ensure modes is a list to simplify processing
    if not isinstance(modes, list):
        modes = [modes]

    # Read the original netlist file
    with open(original_file_path, 'r') as file:
        lines = file.readlines()

    # Process and modify lines
    modified_lines = []
    in_parameters_block = False
    block_lines = []

    for line in lines:
        # Check if the line contains the start of the parameters block
        if '//start parameters' in line:
            in_parameters_block = True
            block_lines.append(line)
            continue

        # Check if the line contains the end of the parameters block
        if '//end parameters' in line:
            in_parameters_block = False
            block_lines.append(line)

            # Process the collected lines in the block
            for block_line in block_lines:
                parts = block_line.split()

                # Update Vdc values if requested
                # if isinstance(X_vec, (list, tuple, np.ndarray)):
                #     iterable = True
                # else:
                #     iterable = False
                #     X_vec = [X_vec]  # Convert scalar to list for uniform processing
                if 'Vdc' in modes:
                    for i, vdc_value in enumerate(X_vec, start=1):
                        parts = [f'Vdc{i}={vdc_value}' if part.startswith(f'Vdc{i}=') else part for part in parts]

                # Update Inudge value if requested
                if 'Inudge' in modes:
                    parts = update_inudge_values(parts, node_to_inudge, outputs, losses, beta)

                # Update deltaR values if requested
                if 'deltaR' in modes:
                    for update in cond_update:
                        res_name = update['resistor']  # Get the resistor name
                        delta = update['cond_update']  # Get the corresponding delta value
                        for i, part in enumerate(parts):
                            if part.startswith(res_name + '='):
                                original_value = float(part.split('=')[1])
                                original_cond = 1/original_value
                                ccond_update = gamma/beta * delta
                                final_cond = original_cond + ccond_update
                                new_value = np.round(1 / final_cond,2)
                                r_update=new_value-original_value
                                # if res_name == "res1" or res_name == "res3":
                                #     new_value = original_value
                                #update =  gamma * (original_value * delta)/(1/original_value + delta)
                                #new_value = original_value + update
                                # if r_update < -100:
                                #      new_value = original_value - 100
                                if new_value < 5:
                                    new_value = original_value
                                # if new_value < 5:
                                #     new_value = 5
                                # if new_value > 10000:
                                #     new_value = original_value
                                    
                                print(f"the resistance change of {part} is {new_value - original_value}")
                                parts[i] = f'{res_name}={new_value}'
                                break  # Found and updated resistor, move to next

                modified_lines.append(' '.join(parts) + '\n')

            # Clear block_lines after processing
            block_lines = []
            continue

        # If in parameters block, add line to block_lines
        if in_parameters_block:
            block_lines.append(line)
        else:
            modified_lines.append(line)

    # Write the modified content to a new file
    with open(new_file_path, 'w') as file:
        file.writelines(modified_lines)

    print(f'Modified netlist saved to {new_file_path}')

# def process_parameters_block(current_parameters, X_vec, Y_vec, cond_update, modes, beta, losses, outputs, node_to_inudge):
#     # Join all parts of a parameter block and split into individual parts
#     full_line = ' '.join(current_parameters).replace('\\', '')  # Remove line continuation for processing
#     parts = full_line.split()

#     # Modify parts as needed
#     for i, part in enumerate(parts):
#         if any(part.startswith(f'{mode}') for mode in modes):
#             if 'Vdc' in modes:
#                 for j, vdc_value in enumerate(X_vec, start=1):
#                     if part.startswith(f'Vdc{j}='):
#                         parts[i] = f'Vdc{j}={vdc_value}'
#             if 'Inudge' in modes:
#                 parts = update_inudge_values(parts, node_to_inudge, outputs, losses, beta)
#             if 'deltaR' in modes:
#                 for update in cond_update:
#                     res_name = update['resistor']
#                     delta = update['cond_update']
#                     if part.startswith(res_name + '='):
#                         original_value = float(part.split('=')[1])
#                         new_value = np.round(original_value + delta,2)
#                         parts[i] = f'{res_name}={new_value}'

#     # Reconstruct the parameter line with line continuations if necessary
#     reconstructed_lines = reconstruct_with_continuations(parts)
#     return reconstructed_lines

# def reconstruct_with_continuations(parts):
#     # Logic to reconstruct lines with appropriate line continuations
#     max_line_length = 80
#     current_line = ''
#     reconstructed_lines = []
#     for part in parts:
#         if len(current_line) + len(part) + 1 > max_line_length:
#             reconstructed_lines.append(current_line + ' \\')
#             current_line = part
#         else:
#             current_line += ' ' + part if current_line else part
#     reconstructed_lines.append(current_line)
#     return [line + '\n' for line in reconstructed_lines]

# Ensure `update_inudge_values` is defined and handles the required logic as well.


def create_node_to_inudge_map(file_path):
    node_to_inudge = {}
    with open(file_path, 'r') as file:
        for line in file:
            if 'isource' in line:
                parts = line.split()
                # The node might be formatted like '(nodeX)', so we need to remove parentheses
                node = parts[2].strip('()')  # Strips off any parentheses
                inudge_key = next((part.split('=')[1] for part in parts if part.startswith('dc=')), None)
                if inudge_key:
                    node_to_inudge[node] = inudge_key
    return node_to_inudge

def update_inudge_values(parts, node_to_inudge, outputs, losses, beta):
    for node, loss in zip(outputs, losses):
        inudge_key = node_to_inudge.get(node)
        if inudge_key:
            i_nudge = np.round(beta * loss,8)
            print(f"I nudge is: {i_nudge}")
            parts = [f'{inudge_key}={i_nudge}' if part.startswith(f'{inudge_key}=') else part for part in parts]
    return parts

def calc_deltaR(voltage_matrix_f, voltage_matrix_n, gamma):
        # Assuming 'resistor' for names and 'deltaV' for voltage differences
    resistor_names = voltage_matrix_f['resistor']
    deltaV_f = voltage_matrix_f['deltaV']  # Correctly access data by field name
    deltaV_n = voltage_matrix_n['deltaV']  # Correctly access data by field name
    cond_update_values= (deltaV_f ** 2 - deltaV_n ** 2)
    cond_update_values=np.round(cond_update_values,6)
        # Define the dtype for the new structured array
    dtype = [('resistor', 'U10'), ('cond_update', 'f8')]
    cond_update = np.empty(len(resistor_names), dtype=dtype)
        # Fill the new structured array
    cond_update['resistor'] = resistor_names
    cond_update['cond_update'] = cond_update_values

    return cond_update
    

    
# def main():
#     input_sample="/home/filip/CMOS130/simulations/upenn/spectre/schematic/netlist/input.scs"
#     new_file_path_nudged="/home/filip/CMOS130/simulations/upenn/spectre/schematic/netlist/input_modified.scs"
#     X_vec=np.array([1, 2])
#     Y_vec=1
#     output_nodes=['node2', 'node3']
#     node_to_inudge={'node5)': 'inudge2', 'node2)': 'inudge1'}
#     cond_update=[('res16', -3. ), ('res15', -0.3), ('res14', -0.3), ('res10', -0.1),
#            ('res11', -0. ), ('res12', -4.2), ('res9', -2.9), ('res13', -0.5),
#            ('res8', -2.9), ('res7', -0.5), ('res6', -1.8), ('res2', -0.1),
#            ('res3', -0.2), ('res4', -2.1), ('res1', -0.3), ('res5', -0.1)]
#     modes=["deltaR"]
#     beta=0.1
#     losses=0.2
#     modify_netlist_general(input_sample, new_file_path_nudged, X_vec, Y_vec, cond_update, modes, beta, losses, output_nodes, node_to_inudge)
    
    
if __name__ == "__main__":
    init_weights=[]
    for i in range(1, 17):
        init_weights.append({'resistor': f'res{i}', 'init_weight': 100})
    sample_file_path='/home/filip/CMOS130/simulations/sample_files/input_upenn.scs'
    new_sample_file_path='/home/filip/CMOS130/simulations/sample_files/input_upenn2.scs'
    modify_input_config(sample_file_path, new_sample_file_path, init_weights)


