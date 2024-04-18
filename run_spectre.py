# import argparse
# import os
# import subprocess

# def run_spectre_simulation(netlist_file_path, output_directory, i):
#     # Ensure the output directory exists
#     os.makedirs(output_directory, exist_ok=True)

#     # Construct the Spectre command with appropriate arguments
#     spectre_command = [
#         "spectre", netlist_file_path, "-64",
#         "+escchars",
#         "-format", "psfascii",
#         "-raw", output_directory,
#         "+lqtimeout", "900",
#         "-maxw", "5",
#         "-maxn", "5",
#         "+logstatus"
#     ]

#     print("Running simulation with command:", ' '.join(spectre_command))

#     # Execute Spectre command
#     try:
#         result = subprocess.run(spectre_command, check=True, text=True, capture_output=True)
#         # Print the output and error
#         print("Simulation output:", result.stdout)
#         print("Iteration", i)
#         if result.stderr:
#             print("Simulation errors:", result.stderr)
#     except subprocess.CalledProcessError as e:
#         # Print the error
#         print(f"Spectre simulation failed: {e}")
# def main():
#     parser = argparse.ArgumentParser(description="Run Spectre Simulation")
#     parser.add_argument('netlist_file_path', type=str, help="Path to the netlist file")
#     parser.add_argument('output_directory', type=str, help="Directory to store simulation outputs")
#     parser.add_argument('iteration', type=int, help="Iteration number")
#     args = parser.parse_args()

#     run_spectre_simulation(args.netlist_file_path, args.output_directory, args.iteration)

# if __name__ == "__main__":
#     main()


import argparse
import subprocess
import os

def run_spectre_simulation(netlist_file_path, output_directory, i, mode):
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)
    # log_file_path = os.path.join(output_directory, 'simulation.log')
    # Construct the Spectre command with appropriate arguments
    spectre_command = [
        "spectre", netlist_file_path, "-64",
        "+escchars",
        "-format", "psfascii",
        "-raw", output_directory,
        # "-log", log_file_path,
        "+lqtimeout", "900",
        "-maxw", "5",
        "-maxn", "5",
        "+logstatus"
    ]

    # Print a message indicating which iteration is running
    print(f"Running {mode} simulation for iteration {i + 1}...")

    # Execute Spectre command
    try:
        # Redirect standard output and error to suppress them
        with open(os.devnull, 'w') as devnull:
            result = subprocess.run(spectre_command, check=True, stdout=devnull, stderr=subprocess.PIPE, text=True)
            # Check if there are errors and print them
            if result.stderr:
                print("Simulation errors during iteration", i + 1, ":", result.stderr)
    except subprocess.CalledProcessError as e:
        # Print the error
        print(f"Spectre simulation failed during iteration {i + 1}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run Spectre Simulation")
    parser.add_argument('netlist_file_path', type=str, help="Path to the netlist file")
    parser.add_argument('output_directory', type=str, help="Directory to store simulation outputs")
    parser.add_argument('iteration', type=int, help="Iteration number")
    args = parser.parse_args()

    run_spectre_simulation(args.netlist_file_path, args.output_directory, args.iteration)

if __name__ == "__main__":
    main()

