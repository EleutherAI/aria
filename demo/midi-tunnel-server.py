import rtmidi
import socket
import time
import struct
import argparse

class MIDIRouter:
    def __init__(self, midi_port="14:0", udp_port=5004):
        self.midi_in = rtmidi.MidiIn()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_port = udp_port
        
        # Print available ports
        ports = self.midi_in.get_ports()
        print(f"Available MIDI ports: {ports}")
        
        # Find and open MIDI port
        for i, port in enumerate(ports):
            if midi_port in port:
                print(f"Opening MIDI port {i}: {port}")
                self.midi_in.open_port(i)
                break
        else:
            print(f"Warning: Could not find port containing '{midi_port}'")
        
        self.midi_in.set_callback(self._midi_callback)

    def _midi_callback(self, message, timestamp):
        try:
            print(f"Received MIDI message: {message[0]}")
            midi_data = struct.pack(f'B' * len(message[0]), *message[0])
            self.socket.sendto(midi_data, ('localhost', self.udp_port))
            print(f"Sent {len(midi_data)} bytes to localhost:{self.udp_port}")
        except Exception as e:
            print(f"Error in callback: {e}")

    def start(self):
        print(f"Routing MIDI messages through SSH tunnel on port {self.udp_port}...")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        print("Shutting down...")
        self.midi_in.close_port()
        self.socket.close()

def parse_args():
    parser = argparse.ArgumentParser(description='MIDI to UDP router')
    parser.add_argument('-midi_p', type=str, default="14:0",
                      help='MIDI port identifier (default: 14:0)')
    parser.add_argument('-udp_p', type=int, default=5004,
                      help='UDP port for forwarding (default: 5004)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    router = MIDIRouter(midi_port=args.midi_p, udp_port=args.udp_p)
    router.start()
