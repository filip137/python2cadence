from psf_utils import PSF, Quantity
import numpy as np



def create_resistor_list(input_file_path, save_as_new=True):

    with open(input_file_path, 'r') as input_file:
        lines = input_file.readlines()
    resistors_list = []
    resistances = {}
    
    # Extract the resistance values from the parameters line
    for line in lines:
        if line.startswith('parameters'):
            params = line.split()
            for param in params:
                if param.startswith('res'):
                    key, val = param.split('=')
                    resistances[key] = val.strip()

    # Parse each line to extract resistor details
    for line in lines:
        if 'resistor ' in line:
            parts = line.split(' ')
            name = parts[0]

            # Extracting node voltages
            node_voltage1 = parts[1].replace('(', '').replace(')', '')
            node_voltage2 = parts[2].replace('(', '').replace(')', '')

            # Extracting resistance key and looking up its value
            resistance_key = next((part.split('=')[1] for part in parts if part.startswith('r=')), "Unknown").strip()
            resistance_value = resistances.get(resistance_key, "Unknown")  # Step 3: Match and assign values

            # Creating the dictionary for the current resistor
            resistor_dict = {
                "name": name,  # This is the label in the netlist, like R4, R3...
                "node voltages": (node_voltage1, node_voltage2),
                "current": "Unknown",  # Maybe useful for later
                "resistance": resistance_value,  # Actual resistance value assigned
                "resistance_key": resistance_key  # Keeping the resistance_key for reference
            }
            resistors_list.append(resistor_dict)

    return resistors_list



def read_and_store_results(psf_ascii_results, resistors_list):
    psf = PSF(psf_ascii_results)  # Assuming PSF is a defined class for handling PSF ASCII files
    dtype = [('resistor', 'U10'), ('V1', 'f8'), ('V2', 'f8'), ('deltaV', 'f8')]
    voltages_matrix = np.empty((0, 4), dtype=dtype) #this would need to be modified for transistors
    for resistor in resistors_list:
        node_voltages = resistor['node voltages']
        name = resistor['resistance_key']
        # Check if the first node voltage is "0" and set V1 accordingly
        if node_voltages[0] == "0":
            V1 = 0
        else:
            V1 = psf.get_signal(node_voltages[0]).ordinate.real  # The last value in the array

        # Similarly, check if the second node voltage is "0" and set V2 accordingly
        if node_voltages[1] == "0":
            V2 = 0
        else:
            V2 = psf.get_signal(node_voltages[1]).ordinate.real

        deltaV = V1 - V2
        new_row = np.array([(name, V1, V2, deltaV)], dtype=dtype)
        voltages_matrix = np.append(voltages_matrix, new_row)

    return voltages_matrix


def loss_function(psf_ascii_results, Y_vec, outputs):
    psf = PSF(psf_ascii_results)
    losses=[]
    for output in outputs:
        try:
            outputV = psf.get_signal(output).ordinate.real
            # Assuming Y_vec is either an array or a list with the same length as outputs
            if isinstance(Y_vec, (list, np.ndarray)):
                loss = outputV - Y_vec[outputs.index(output)]
            else:
                loss = outputV - Y_vec
            losses.append(loss)
        except:
            print(f"Error, no node is named {output}")
            losses.append(np.nan)  # Use np.nan to handle errors but keep the array operations valid

    # Convert list of losses to a numpy array
    losses = np.array(losses)

    # Return the mean loss ignoring NaNs
    return losses  # This computes the mean while ignoring NaN values

def calc_deltaR(voltage_matrix_f, voltage_matrix_n, beta):
        # Assuming 'resistor' for names and 'deltaV' for voltage differences
    resistor_names = voltage_matrix_f['resistor']
    deltaV_f = voltage_matrix_f['deltaV']  # Correctly access data by field name
    deltaV_n = voltage_matrix_n['deltaV']  # Correctly access data by field name
    cond_update_values = beta * 1/(deltaV_f ** 2 - deltaV_n ** 2) ##sorted
    cond_update_values=np.round(cond_update_values,1)
        # Define the dtype for the new structured array
    dtype = [('resistor', 'U10'), ('cond_update', 'f8')]
    cond_update = np.empty(len(resistor_names), dtype=dtype)
        # Fill the new structured array
    cond_update['resistor'] = resistor_names
    cond_update['cond_update'] = cond_update_values

    return cond_update
    ## here the result file comes from the psf_reader
    ##with open(results_file_path, 'r') as results_file:
      ##  result_lines = results_file.readlines()

    ##node_voltages = {}

    ##for line in result_lines:
      ##  if not line.startswith('#'):
        ##    parts=line.split()
          ##  node_voltages[parts[0]]=parts[1]




    ##for resistor in resistors_list:
            # Unpack the current node_voltages tuple to individual nodes
      ##  node1, node2 = resistor["node voltages"]

            # Update the node_voltages with values from the node_voltages dictionary
            # The get method returns the existing value if the node is found, otherwise it keeps the original value
        ##updated_voltage1 = node_voltages.get(node1, node1)  # Default to node1 if not found
        ##updated_voltage2 = node_voltages.get(node2, node2)  # Default to node2 if not found

            # Update the resistor's node_voltages with the new values
        ##resistor["node voltages"] = (updated_voltage1, updated_voltage2)

    ##return resistors_list

    # Assuming node_voltages is your dictionary with the correct voltages and resistors_list is your current list of resistors
    ##resistors_list_updated = update_resistor_voltages(resistors_list, node_voltages)

    # You can then print or further process resistors_list_updated as needed
    ##print(resistors_list_updated)
    # Print or return the list of dictionaries


