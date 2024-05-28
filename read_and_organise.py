from psf_utils import PSF, Quantity
import numpy as np
import logging


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
                "node_voltage_ter1": None,
                "node_voltage_ter2": None,
                "current": "Unknown",  # Maybe useful for later
                "resistance": resistance_value,  # Actual resistance value assigned
                "resistance_key": resistance_key  # Keeping the resistance_key for reference
            }
            resistors_list.append(resistor_dict)

    return resistors_list

def create_memresistor_array(resistors_list):
    
    
    dtype = [('resistor', 'U10'), ('voltage1', 'float32'), ('voltage2', 'float32')]
    structured_data = []
    
    for resistor in resistors_list:
        # Extract name and voltages, assume some default values if necessary
        name = resistor['resistance_key']
        voltage1 = resistor.get('node_voltage_ter1', 0.0)  # Default voltage 0.0 if not available
        voltage2 = resistor.get('node_voltage_ter2', 0.0)  # Default voltage 0.0 if not available
        
        # Append a tuple for each resistor to the list
        structured_data.append((name, voltage1, voltage2))
    
    # Create a structured array from the list of tuples
    resistor_array = np.array(structured_data, dtype=dtype)
    return resistor_array


def update_resistor_list(resistors_list, node_voltage_dict):
    # Iterate over each resistor in the list
    for resistor in resistors_list:
        # Extract the node voltages from each resistor
        node1, node2 = resistor['node voltages']

        # Fetch the voltage values from the node_voltage_dict
        # and update the resistor's node_voltage_values if the node exists in node_voltage_dict
        voltage1 = node_voltage_dict.get(node1, "Unknown")  # Defaults to "Unknown" if node is not found
        voltage2 = node_voltage_dict.get(node2, "Unknown")  # Defaults to "Unknown" if node is not found

        # Update the node_voltage_values in the resistor dictionary
        resistor['node_voltage_ter1'] = voltage1
        resistor['node_voltage_ter2'] = voltage2

    # Optionally return the updated list if needed for further operations
    return resistors_list



def create_node_voltage_dict(resistors_list):
    node_voltage_dict = {}

    # Extract unique node voltages and initialize their values
    for resistor in resistors_list:
        for node_voltage in resistor['node voltages']:
            # Initialize each unique node voltage with a placeholder for its future voltage value
            if node_voltage not in node_voltage_dict:
                node_voltage_dict[node_voltage] = None  

    return node_voltage_dict


def read_node_voltages(psf_ascii_results, node_voltage_dict):
    
    # Reads all the node voltages that are in the node_voltage_dict
    
    psf = PSF(psf_ascii_results)
    errors = {}

    for node in node_voltage_dict:
        try:
            output = psf.get_signal(node).ordinate.real
            node_voltage_dict[node] = output
        except Exception as e:
            print(f"Error retrieving data for node '{node}': {e}")

            # Handle the error by setting the node's output to NaN
            # Determine a sensible default size if possible or a single NaN
            if isinstance(node_voltage_dict[node], np.ndarray):
                default_size = len(node_voltage_dict[node])
            else:
                default_size = 1  # Adjust based on expected data size
            node_voltage_dict[node] = np.full(default_size, np.nan)

            # Record the error for possible further analysis
            errors[node] = str(e)

    # Print errors if any occurred
    if errors:
        print(f"Encountered errors with nodes: {errors}")

    return node_voltage_dict


def read_and_store_results(psf_ascii_results, resistors_list):
    psf = PSF(psf_ascii_results)  
    dtype = [('resistor', 'U10'), ('V1', 'f8'), ('V2', 'f8'), ('deltaV', 'f8')]
    voltages_matrix = np.empty((0, 4), dtype=dtype) #this would need to be modified for transistors
    for resistor in resistors_list:
        node_voltages = resistor['node voltages']
        name = resistor['resistance_key']
        # Check if the first node voltage is 0 and set V1 accordingly
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






def read_all_results(psf_ascii_results, nodes):
    psf = PSF(psf_ascii_results)
    
    # We'll use a list to collect data because we don't know the number of outputs yet
    results = []
    
    for node in nodes:
        try:
            # Assuming get_signal returns a consistent-sized array or list for each node
            output = psf.get_signal(node).ordinate.real
            results.append(output)
        except Exception as e:
            # Print an error message indicating which node had an issue
            print(f"Error retrieving data for node '{node}': {e}")
            # Handle missing data; size needs to match other node outputs, use np.nan to fill if unknown
            if results:
                # Assume all nodes should have the same length output, fill with NaNs if an error occurs
                results.append(np.full(len(results[0]), np.nan))
            else:
                # If no successful data fetch yet, we cannot assume length, handle differently if needed
                results.append(np.array([np.nan]))  # Placeholder, adjust as needed

    # Convert the list of outputs into a 2D NumPy array with each node's outputs as columns
    results_matrix = np.column_stack(results)
    return results_matrix
            
def loss_function(psf_ascii_results, Y_vec, outputs): ## can be improved to not open results files again
    psf = PSF(psf_ascii_results)
    losses=[]
    for output in outputs:
        try:
            outputV = psf.get_signal(output).ordinate.real
            # Assuming Y_vec is either an array or a list with the same length as outputs
            if isinstance(Y_vec, (list, np.ndarray)):
                loss = Y_vec[outputs.index(output)] - outputV
            else:
                loss = Y_vec - outputV  #target - measured
            losses.append(loss)
        except:
            print(f"Error, no node is named {output}")
            losses.append(np.nan)  # Use np.nan to handle errors but keep the array operations valid

    # Convert list of losses to a numpy array
    losses = np.array(losses)

    # Return the mean loss
    return losses  


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


