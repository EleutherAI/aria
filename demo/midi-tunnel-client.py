import socket
import rtmidi
import time
import subprocess
import signal
import sys
import os
import argparse

SSH_SERVER = "home-4090.remote"
def parse_arguments():
    parser = argparse.ArgumentParser(description='MIDI UDP bridge with SSH tunnel')
    parser.add_argument('-p', '--port', type=int, default=5004,
                      help='UDP port number (default: 5004)')
    return parser.parse_args()

def kill_existing_process(port):
    # Check and kill existing process on remote server
    check_command = f"ssh {SSH_SERVER} 'lsof -ti :{port}'"
    try:
        pid = subprocess.check_output(check_command, shell=True).decode().strip()
        if pid:
            print(f"Found existing process {pid} on port {port}, killing it...")
            kill_command = f"ssh {SSH_SERVER} 'kill -9 {pid}'"
            subprocess.run(kill_command, shell=True)
            # Wait a moment for the port to be freed
            time.sleep(1)
    except subprocess.CalledProcessError:
        # No existing process found
        pass

def setup_ssh_tunnel(port):
    while True:
        try:
            # Kill any existing process first
            kill_existing_process(port)

            # Start SSH tunnel using socat
            print(f"Attempting to establish SSH tunnel on port {port}...")
            ssh_command = f"ssh {SSH_SERVER} 'socat -u UDP4-RECV:{port} STDOUT'"
            local_socat = f"socat -u STDIN UDP4-SEND:localhost:{port}"

            ssh_process = subprocess.Popen(ssh_command, shell=True, stdout=subprocess.PIPE)
            socat_process = subprocess.Popen(local_socat, shell=True, stdin=ssh_process.stdout)

            # Check if the processes started successfully
            time.sleep(1)
            if ssh_process.poll() is not None:  # Process terminated
                raise subprocess.CalledProcessError(ssh_process.returncode, ssh_command)
            
            print("SSH tunnel established successfully!")
            return ssh_process, socat_process

        except (subprocess.CalledProcessError, OSError) as e:
            print(f"Failed to establish SSH tunnel: {str(e)}")
            print("Retrying in 1 second...")
            time.sleep(1)

def create_virtual_port(port):
    midi_out = rtmidi.MidiOut()
    # Create a virtual MIDI port with port number in name
    midi_out.open_virtual_port(f"UDP_{port}")
    return midi_out

def start_udp_listener(port):
    # Create UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('localhost', port))
    return sock

def split_midi_messages(data):
    """Split a byte array into individual MIDI messages."""
    messages = []
    data_list = list(data)
    i = 0
    while i < len(data_list):
        # Check if we have a status byte (most significant bit is 1)
        if data_list[i] >= 0x80:
            # Most MIDI messages are 3 bytes
            if i + 2 < len(data_list):
                messages.append(data_list[i:i+3])
                i += 3
            else:
                # Handle incomplete message at end of buffer
                break
        else:
            # Skip non-status bytes (shouldn't happen in properly formatted MIDI)
            i += 1
    return messages

def cleanup(ssh_process, socat_process, midi_out, sock):
    print("\nCleaning up...")
    # Kill the SSH and socat processes
    if ssh_process:
        os.killpg(os.getpgid(ssh_process.pid), signal.SIGTERM)
    if socat_process:
        socat_process.terminate()
    # Close MIDI and socket
    if midi_out:
        midi_out.close_port()
    if sock:
        sock.close()

def main():
    args = parse_arguments()
    port = args.port

    ssh_process = None
    socat_process = None
    midi_out = None
    sock = None

    try:
        # Setup SSH tunnel first
        print(f"Setting up SSH tunnel on port {port}...")
        ssh_process, socat_process = setup_ssh_tunnel(port)

        # Setup MIDI and UDP
        print(f"Creating virtual MIDI port UDP_{port}...")
        midi_out = create_virtual_port(port)
        print(f"Starting UDP listener on port {port}...")
        sock = start_udp_listener(port)

        print(f"UDP MIDI Bridge started - listening on port {port}")

        while True:
            data, addr = sock.recvfrom(1024)
            if data:
                # Split the data into individual MIDI messages
                midi_messages = split_midi_messages(data)
                for midi_message in midi_messages:
                    print(f"Sending MIDI message: {midi_message}")
                    midi_out.send_message(midi_message)

    except KeyboardInterrupt:
        print("\nShutting down UDP MIDI Bridge...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cleanup(ssh_process, socat_process, midi_out, sock)

if __name__ == "__main__":
    main()
