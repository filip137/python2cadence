import subprocess
import os
import time

def run_spectre_simulation(netlist_file_path, output_directory, i, mode):
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Construct the Spectre command with appropriate arguments
    spectre_command = [
        "spectre", netlist_file_path, "-64",
        "+escchars",
        "-format", "psfascii",
        "-raw", output_directory,
        "+lqtimeout", "900",
        "-maxw", "5",
        "-maxn", "5",
        "+logstatus"
    ]

    # Print a message indicating which iteration is running
    print(f"Running {mode} simulation for iteration {i + 1}...")

    # Start timing the Python command construction and start process
    start_python_time = time.time()

    # Initialize timing variables
    spectre_start_time = None
    spectre_end_time = None

    # Execute Spectre command
    try:
        # Start timing the Spectre process execution
        spectre_start_time = time.time()
        result = subprocess.run(spectre_command, check=True)
        #result = subprocess.run(spectre_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        spectre_end_time = time.time()
        
    except subprocess.CalledProcessError as e:
        # Print the error
        print(f"Spectre simulation failed during iteration {i + 1}: {e}")
        spectre_end_time = time.time()  # End timing even if there is an error
    finally:
        # End timing the Python command construction and start process
        end_python_time = time.time()

        # Calculate and print the elapsed times
        if spectre_start_time is not None and spectre_end_time is not None:
            python_execution_time = spectre_start_time - start_python_time
            spectre_execution_time = spectre_end_time - spectre_start_time
            total_time = end_python_time - start_python_time

            print(f"Time to construct and start Spectre command: {python_execution_time:.2f} seconds")
            print(f"Spectre process execution time: {spectre_execution_time:.2f} seconds")
            print(f"Total time for iteration {i + 1}: {total_time:.2f} seconds")
        else:
            print("Timing information is incomplete due to an error during execution.")

def run_simulation():
    netlist_file_path = "/home/filip/CMOS130/simulations/sample_files/upenn_classification.scs"
    output_directory = "/home/filip/CMOS130/simulations/various_tests/upenn"
    iteration = 0  # You can change this to the desired iteration number
    mode = "classification"  # You can change this to the desired mode

    run_spectre_simulation(netlist_file_path, output_directory, iteration, mode)

if __name__ == "__main__":
    run_simulation()
    