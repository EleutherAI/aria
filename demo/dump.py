import mido
import sys
from mido import tick2second, second2tick

# Check if correct number of arguments is provided
if len(sys.argv) != 3:
    print("Usage: python script.py <input_midi_file> <target_seconds>")
    sys.exit(1)

# Get command line arguments
input_file = sys.argv[1]
target_seconds = float(sys.argv[2])

try:
    mid = mido.MidiFile(input_file)
except Exception as e:
    print(f"Error loading MIDI file: {e}")
    sys.exit(1)

curr_tick = 0
idx = 0
tempo = None

# First get the tempo
for msg in mid.tracks[0]:
    if msg.type == "set_tempo":
        tempo = msg.tempo
        break

print(f"Found tempo: {tempo}")

# Then find the right index
curr_tick = 0
for idx, msg in enumerate(mid.tracks[0]):
    curr_tick += msg.time
    seconds = tick2second(
        tick=curr_tick,
        ticks_per_beat=mid.ticks_per_beat,
        tempo=tempo,
    )
    print(f"At index {idx}, time: {seconds:.2f} seconds")
    if seconds > target_seconds:
        print(f"Breaking at index {idx}")
        break

print(f"Inserting at index {idx}")

# Insert the messages at the found index
mid.tracks[0].insert(
    idx,
    mido.Message(
        type="control_change",
        control=66,
        value=127,
        time=0,
    ),
)
mid.tracks[0].insert(
    idx + 1,
    mido.Message(
        type="control_change",
        control=66,
        value=0,
        time=second2tick(
            second=0.01,
            ticks_per_beat=mid.ticks_per_beat,
            tempo=tempo,
        ),
    ),
)

# Generate output filename based on input filename
output_path = "/home/loubb/Dropbox/shared/test.mid"
mid.save(output_path)
print(f"Saved modified MIDI file to: {output_path}")
