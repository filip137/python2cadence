import subprocess
import threading
import time
from queue import Queue, Empty




def start_eldo_simulation(sample_file):
    """Starts the Eldo simulation subprocess in interactive mode."""
    try:
        eldo_command = f"eldo {sample_file} -inter"
        process = subprocess.Popen(
            eldo_command, 
            shell=True, 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True,
            bufsize=1  # Line buffered
        )
        return process
    except subprocess.CalledProcessError as e:
        print(f"Eldo simulation failed: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def send_command_to_eldo(process, command):
    if process:
        process.stdin.write(command + "\n")
        process.stdin.flush()

def read_eldo_output_once(process):
    output = []
    while True:
        line = process.stdout.readline().strip()
        if line == '' and process.poll() is not None:
            break
        if line:
            output.append(line)
        if "eldo>" in line:  # Assuming "eldo>" is the prompt indicating end of output
            break
    return output

def extract_results(process, mode, node_voltages, resistor_value_dict):
    if mode == "get_resistance":
        for resistor in resistor_value_dict.keys():
            eldo_command = f"PRINT P({resistor})"
            send_command_to_eldo(process, eldo_command)
            output_lines = read_eldo_output_once(process)
            for line in output_lines:
                if resistor in line:
                    new_resistance = line.split("=")[1].strip()
                    resistor_value_dict[resistor] = float(new_resistance)
                    break  # Exit the loop once the value is found
            else:
                print(f"Error retrieving resistance for {resistor}: {output_lines}")
        return resistor_value_dict

    if mode == "get_voltage":
        for node in node_voltages.keys():
            eldo_command = f"PRINT V({node})"
            send_command_to_eldo(process, eldo_command)
            output_lines = read_eldo_output_once(process)
            for line in output_lines:
                if "=" in line:
                    new_voltage = line.split("=")[1].strip()
                    node_voltages[node] = float(new_voltage)
                    break  # Exit the loop once the value is found
            else:
                print(f"Error retrieving voltage for {node}: {output_lines}")
        return node_voltages

# Example usage
sample_file = "/home/filip/CMOS130/simulations/sample_files/eldo_samples/6transistor_sample/6_resistors.cir"
output_list = []

# Start the Eldo simulation
eldo_process = start_eldo_simulation(sample_file)

# Allow some time for the initial output
time.sleep(2)

# Extract results
node_voltages = {"NET1": 0.0, "NET7": 0.0}
resistor_value_dict = {"RES1": 0.0, "RES3": 0.0}
updated_resistor_values = extract_results(eldo_process, "get_resistance", node_voltages, resistor_value_dict)
updated_node_voltages = extract_results(eldo_process, "get_voltage", node_voltages, resistor_value_dict)

# Print the updated values
print("Updated resistor values:", updated_resistor_values)
print("Updated node voltages:", updated_node_voltages)

# Stop the Eldo process
eldo_process.terminate()

print("Interaction completed.")
