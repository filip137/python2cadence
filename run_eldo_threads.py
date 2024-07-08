import argparse
import subprocess
import os
import threading
import time
from queue import LifoQueue, Empty

def enqueue_output(out, queue, stream_name="stdout"):
    """Worker thread function to read lines from subprocess stdout or stderr and put them into a LifoQueue."""
    for line in iter(out.readline, ''):
        queue.put((stream_name, line))
    out.close()

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
    """Sends a command to the Eldo subprocess."""
    if process:
        process.stdin.write(command + "\n")
        process.stdin.flush()

def set_eldo_simulation(process, mode, input_values, resistor_value_dict,  inudge_dict = None):
    """Sets the simulation parameters for the Eldo process."""
    try:
        if mode == "free":
            for first_entry, second_entry in input_values:
                eldo_command = f"SET P ({first_entry}) = {second_entry}"
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

def run_eldo_simulation(process):
    """Runs the Eldo simulation."""
    try:
        send_command_to_eldo(process, "RUN")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def read_eldo_output(queue):
    """Reads the last line from the LifoQueue."""
    try:
        stream_name, line = queue.get_nowait()
        return line.strip()
    except Empty:
        return ""


def extract_results(process, queue, mode, node_voltages, resistor_value_dict):
    if mode == "get_resistance":
        for resistor in resistor_value_dict.keys():
            eldo_command = f"PRINT P({resistor})"
            send_command_to_eldo(process, eldo_command)
            output = read_eldo_output(queue)
            if "=" in output:
                new_resistance = output.split("=")[1].strip()
                resistor_value_dict[resistor] = float(new_resistance)
            else:
                print(f"Error retrieving resistance for {resistor}: {output}")
        return resistor_value_dict

    if mode == "get_voltage":
        for node in node_voltages.keys():
            eldo_command = f"PRINT V({node})"
            send_command_to_eldo(process, eldo_command)
            output = read_eldo_output(queue)
            if "=" in output:
                new_voltage = output.split("=")[1].strip()
                node_voltages[node] = float(new_voltage)
            else:
                print(f"Error retrieving voltage for {node}: {output}")
        return node_voltages
def main():
    sample_file = "/home/filip/CMOS130/simulations/sample_files/eldo_samples/6transistor_sample/6_resistors.cir"

    # Start the Eldo simulation
    eldo_process = start_eldo_simulation(sample_file)

    if not eldo_process:
        return

    # Create LifoQueues to hold stdout and stderr lines
    stdout_queue = LifoQueue()
    stderr_queue = LifoQueue()

    # Start threads to read stdout and stderr
    stdout_thread = threading.Thread(target=enqueue_output, args=(eldo_process.stdout, stdout_queue, "stdout"))
    stderr_thread = threading.Thread(target=enqueue_output, args=(eldo_process.stderr, stderr_queue, "stderr"))
    stdout_thread.daemon = True
    stderr_thread.daemon = True
    stdout_thread.start()
    stderr_thread.start()
    
    # Allow some time for the initial output
    time.sleep(2)
    input_values=[("VDC1", "1"), ("VDC2", "2")]
    resistor_value_dict = {"RES1": 0.0, "RES2": 0.0, "RES3" : 1.0, "RES4" : 2.0, "RES5" : 3.0, "RES6" : 4.0}

    # Set simulation parameters
    set_eldo_simulation(eldo_process, "free", input_values, resistor_value_dict)
    set_eldo_simulation(eldo_process, "set_resistances", input_values, resistor_value_dict)

    # Allow some time for the command to be processed
    time.sleep(2)

    # Run the Eldo simulation
    run_eldo_simulation(eldo_process)

    # Allow some time for the simulation to run
    time.sleep(2)

    # Extract results
    node_voltages = {"NET1": 0.0, "NET2": 0.0, "NET3": 0.0, "NET7" : 0.0}

    updated_resistor_values = extract_results(eldo_process, stdout_queue, "get_resistance", node_voltages, resistor_value_dict)
    updated_node_voltages = extract_results(eldo_process, stdout_queue, "get_voltage", node_voltages, resistor_value_dict)

    # Print the updated values
    print("Updated resistor values:", updated_resistor_values)
    print("Updated node voltages:", updated_node_voltages)

    # Check for any errors in stderr
    while True:
        try:
            stream_name, line = stderr_queue.get_nowait()
            print(f"Error from {stream_name}: {line.strip()}")
        except Empty:
            break

    # Stop the Eldo process
    eldo_process.terminate()

    print("Interaction completed.")

if __name__ == "__main__":
    main()



