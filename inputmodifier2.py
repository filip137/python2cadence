import numpy as np
from read_and_organise import *
def modify_netlist_general(original_file_path, new_file_path, X_vec, Y_vec, cond_update, modes, beta, losses, outputs, node_to_inudge):
    # Ensure modes is a list to simplify processing
    if not isinstance(modes, list):
        modes = [modes]

    # Read the original netlist file
    with open(original_file_path, 'r') as file:
        lines = file.readlines()

    # Process and modify lines
    modified_lines = []
    for line in lines:
        if line.strip().startswith('parameters'):
            parts = line.split()

            # Update Vdc values if requested
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
                            new_value = original_value + delta
                            parts[i] = f'{res_name}={new_value}'
                            break  # Found and updated resistor, move to next

            line = ' '.join(parts) + '\n'

        modified_lines.append(line)

    # Write the modified content to a new file
    with open(new_file_path, 'w') as file:
        file.writelines(modified_lines)

    print(f'Modified netlist saved to {new_file_path}')

def create_node_to_inudge_map(file_path):
    node_to_inudge = {}
    with open(file_path, 'r') as file:
        for line in file:
            if 'isource' in line:
                parts = line.split()
                node = parts[2]  # Adjust this index based on your netlist format
                inudge_key = next((part.split('=')[0] for part in parts if part.startswith('dc=')), None)
                if inudge_key:
                    node_to_inudge[node] = inudge_key
    return node_to_inudge    


def update_inudge_values(parts, node_to_inudge, outputs, losses, beta):
    for node, loss in zip(outputs, losses):
        inudge_key = node_to_inudge.get(node)
        if inudge_key:
            i_nudge = beta * loss
            parts = [f'{inudge_key}={i_nudge}' if part.startswith(f'{inudge_key}=') else part for part in parts]
    return parts
    

    
def main():
    original_file_path="/home/filip/CMOS130/simulations/resistors/spectre/schematic/netlist/input.scs"
    new_file_path="/home/filip/CMOS130/simulations/input_files/inp21.scs"
    X_vec=np.array([1, 2])
    Y_vec=1
    cond_update=1
    modes=["Vdc"]
    beta=0.1
    loss=0.2
    modify_netlist_general(original_file_path, new_file_path, X_vec, Y_vec, cond_update, modes, beta, loss)
    
    
if __name__ == "__main__":
    main()

#new_file_path = '/home/filip/Documents/tryouts/spectre/schematic/netlist/input_modified3.scs'
#X = [1, 2]  # Example array of input voltages for Vdc parameters
#Y = 0.05    # Example scalar value for Inudge
#beta=0.1
#original_file_path="/home/filip/Documents/tryouts/11.4/schematic/netlist/input.scs"
#psf_ascii_results = "/home/filip/Documents/tryouts/11.4/schematic/netlist/input.raw/dcOp.dc"
#resistors_list=read_input_and_organise(input_file_path, save_as_new=True)
#voltages_matrix_f=read_and_store_results(psf_ascii_results, resistors_list)
#voltages_matrix_n=np.copy(voltages_matrix_f)
#voltages_matrix_n['deltaV']+=1

#cond_update=calc_deltaR(voltages_matrix_f, voltages_matrix_n, 0.1)
#loss=loss_function(psf_ascii_results, Y)
#modify_netlist_general(original_file_path, new_file_path, [], 0.05, cond_update, ['Vdc', 'deltaR'], beta, loss)
