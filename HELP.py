import subprocess
import threading
import time
from queue import Queue, Empty

def enqueue_output(out, queue):
    """Worker thread function to read lines from subprocess stdout and put them into a queue."""
    for line in iter(out.readline, ''):
        queue.put(line)
    out.close()


def extract_output_value(output):
    """Extracts the numerical value from the output list."""
    for line in output:
        if '=' in line:
            return line.split('=')[1].strip()
    return None


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

def read_eldo_output(queue):
    """Reads the relevant output from the queue, capturing only lines after the most recent `eldo>` prompt."""
    capturing = False
    output = []
    while True:
        try:
            line = queue.get_nowait()
        except Empty:
            break
        else:
            print(f"Read line: {line.strip()}")  # Debug print
            if "eldo>" in line:
                if capturing:
                    # We encountered the second `eldo>`, stop capturing
                    break
                else:
                    # We encountered the first `eldo>`, start capturing
                    capturing = True
                    output = []  # Clear any previous output
            if capturing:
                output.append(line.strip())
    return output

# Example usage
sample_file = "/home/filip/CMOS130/simulations/sample_files/eldo_samples/eldo_input.cir"

# Start the Eldo simulation
eldo_process = start_eldo_simulation(sample_file)

# Create a queue to hold stdout lines
stdout_queue = Queue()

# Start a thread to read stdout
stdout_thread = threading.Thread(target=enqueue_output, args=(eldo_process.stdout, stdout_queue))
stdout_thread.daemon = True
stdout_thread.start()

# Allow some time for the initial output
time.sleep(2)

# Send first command to Eldo process
send_command_to_eldo(eldo_process, "PRINT P(VDC1)")

# Allow some time for the command to be processed
time.sleep(2)

# Read output from Eldo process
output1 = read_eldo_output(stdout_queue)

# Send second command to Eldo process
send_command_to_eldo(eldo_process, "PRINT P(RES1)")

# Allow some time for the command to be processed
time.sleep(2)

# Read output from Eldo process again
output2 = read_eldo_output(stdout_queue)

# Print the outputs
print(f"Output 1: {output1}")
print(f"Output 2: {output2}")

# Stop the Eldo process
eldo_process.terminate()

print("Interaction completed.")









# Example usage




