import subprocess
import time
import re

import os

def start_eldo_simulation(sample_file, output_dir):
    """Starts the Eldo simulation subprocess in interactive mode, ensuring directory exists."""
    try:
        # Manually set the PATH to include the directory where Eldo is located
        os.environ['PATH'] += ':/cao/Softs/cadence/INNOVUS162/bin'
        os.environ['PATH'] += ':/cao/Softs/cadence/SPECTRE191/tools/bin'

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        eldo_command = ["eldo", sample_file, "-inter", "-createoutpath", output_dir]

        return subprocess.Popen(
            eldo_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
    except Exception as e:
        print(f"Error starting Eldo simulation: {e}")
        return None

def send_command_to_eldo(process, command):
    """Sends a command to the Eldo subprocess, ensuring it's still open."""
    if process.poll() is None:  # None means the process is still running
        print(f"Sending command: {command}")
        try:
            process.stdin.write(command + "\n")
            process.stdin.flush()
        except Exception as e:
            print(f"Error sending command: {e}")
    else:
        print("Cannot send command, subprocess has terminated.")
        
def set_eldo_simulation(process, mode, input_values, resistor_value_dict,  inudge_dict):
    """Sets the simulation parameters for the Eldo process."""
    try:
        if mode == "free":
            for vol_source, vol in input_values.items():
                eldo_command = f"SET P ({vol_source}) = {vol}"
                send_command_to_eldo(process, eldo_command)
            for inudge, curr in inudge_dict.items():
                eldo_command = f"SET P ({inudge}) = 0"
                send_command_to_eldo(process, eldo_command)
        elif mode == "nudge":
            for inudge, curr in inudge_dict.items():
                eldo_command = f"SET P ({inudge}) = {curr}"
                send_command_to_eldo(process, eldo_command)

        elif mode == "set_resistances":
            for res_key, res_value in resistor_value_dict.items():
                eldo_command = f"SET P ({res_key}) = {res_value}"
                send_command_to_eldo(process, eldo_command)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def read_eldo_output(process, stop_here):
    """Reads output from the Eldo subprocess until the prompt appears, storing only the second to last line."""
    last_line = None  # This will store the last line
    second_to_last_line = None  # This will store the second to last line
    
    while True:  # Use a loop to keep reading until the prompt is found
        line = process.stdout.readline().strip()
        if line:
          #  print(f"Reading output: {line}")
            if stop_here in line:  # Check for the prompt indicating ready for next command
                break  # Exit the loop when the prompt is detected
    
    return line



def run_eldo_simulation(process):
    """Runs the Eldo simulation."""
    try:
        send_command_to_eldo(process, "GO")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        
def quit_eldo_simulation(process):
    """QUITS the Eldo simulation."""
    try:
        send_command_to_eldo(process, "QUIT")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        
def extract_results(process, mode, node_voltages, resistor_value_dict):
    if mode == "get_resistance":
        for resistor in resistor_value_dict.keys():
            eldo_command = f"PRINT P({resistor})"
            send_command_to_eldo(process, eldo_command)
            line = read_eldo_output(process, resistor)
           # print(f"Output line for resistor {resistor}: '{line}'")  # Debugging line
            
            # Regular expression pattern to match the resistor value
            pattern = rf"{resistor}\s*=\s*([0-9.eE+-]+)"
            match = re.search(pattern, line)
            
            if match:
                # Extracting the matched value from the capturing group
                new_resistance = match.group(1)
                resistor_value_dict[resistor] = float(new_resistance)
            else:
                print(f"Error retrieving resistance for {resistor}: {line}")
        return resistor_value_dict


    if mode == "get_voltage":
        new_node_voltages = {}
        for node in node_voltages.keys():
            eldo_command = f"PRINT V({node})"
            send_command_to_eldo(process, eldo_command)
            line = read_eldo_output(process, node)
            
            # Regular expression pattern to match the node voltage value
            pattern = rf"{node}\s+([0-9.eE+-]+)"
            match = re.search(pattern, line)
            
            if match:
                # Extracting the matched value from the capturing group
                new_voltage = match.group(1)
                new_node_voltages[node] = float(new_voltage)
            elif node in line:
                new_voltage = line.split("=")[1].strip()
                new_node_voltages[node] = float(new_voltage)
            else:
                print(f"Error retrieving voltage for {node}: {line}")
        return new_node_voltages
    
def wait_for_eldos_completion(process):
    """
    Waits until the specified completion message is found in the process output.
    """
    completion_message = "Eldo interactive runs completed."

    while True:
        line = process.stdout.readline().strip()
        if line:
            print(f"Reading output: {line}")
            if completion_message in line:
             #   print("Completion message detected.")
                break

    
def main():
    sample_file = "/home/filip/CMOS130/simulations/sample_files/eldo_samples/6_resistors.cir"
    output_dir="/home/filip/CMOS130/simulations/simulations"
    # Start the Eldo simulation
    eldo_process = start_eldo_simulation(sample_file, output_dir)
    if not eldo_process:
        return

    # Allow some time for the initial output
    time.sleep(2)
    input_values = {"VDC1": 1, "VDC2": 2}  # Adjusted for dictionary usage
    resistor_value_dict = {"RES1": 1.0, "RES2": 100.0, "RES3": 1.0, "RES4": 2.0, "RES5": 3.0, "RES6": 4.0}
    node_voltages = {"NET1": 0.0, "NET2": 0.0, "NET3": 0.0, "NET7": 0.0}

    # Set simulation parameters
    set_eldo_simulation(eldo_process, "free", input_values, {})  # Corrected input format for set_eldo_simulation
    set_eldo_simulation(eldo_process, "set_resistances", {}, resistor_value_dict)

    # Run the Eldo simulation
    run_eldo_simulation(eldo_process)

    # Extract results
    updated_resistor_values = extract_results(eldo_process, "get_resistance", node_voltages, resistor_value_dict)
    updated_node_voltages = extract_results(eldo_process, "get_voltage", node_voltages, resistor_value_dict)




    input_values = {"VDC1": 1, "VDC2": 2}  # Adjusted for dictionary usage


        # Run the Eldo simulation
    run_eldo_simulation(eldo_process)
    
    
        
    updated_resistor_values = extract_results(eldo_process, "get_resistance", node_voltages, resistor_value_dict)
    updated_node_voltages = extract_results(eldo_process, "get_voltage", node_voltages, resistor_value_dict)

    
    quit_eldo_simulation(eldo_process)
    # Close the Eldo process
    #eldo_process.terminate()
    #eldo_process.wait()

    print("Interaction completed.")

if __name__ == "__main__":
    main()




