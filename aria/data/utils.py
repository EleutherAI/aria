"""Utils for data processing"""

import mido

from collections import defaultdict


# Loosely inspired by pretty_midi src
def _extract_track_data(track: mido.MidiTrack):
    messages = []
    meta_messages = []
    last_note_on = defaultdict(list)

    for message in track:
        # Meta messages
        if message.is_meta is True:
            if message.type == "text" or message.type == "copyright":
                meta_messages.append(
                    {
                        "type": message.type,
                        "data": message.text,
                    }
                )
            elif message.type == "set_tempo":
                messages.append(
                    {
                        "type": message.type,
                        "data": message.tempo,
                        "tick": message.time,
                    }
                )
        # Instrument messages
        elif message.type == "program_change":
            messages.append(
                {
                    "type": "instrument",
                    "data": message.program,
                    "tick": message.time,
                    "channel": message.channel,
                }
            )
        # Pedal messages
        elif message.type == "control_change" and message.control == 64:
            if message.value == 127:
                val = 1
            elif message.value == 0:
                val = 0
            messages.append(
                {
                    "type": "pedal",
                    "data": val,
                    "tick": message.time,
                    "channel": message.channel,
                }
            )
        # Note messages
        elif message.type == "note_on" and message.velocity > 0:
            last_note_on[message.note].append(message.time)
        elif message.type == "note_off" or (
            message.type == "note_on" and message.velocity == 0
        ):
            # Ignore non-existent note-ons
            if message.note in last_note_on:
                end_tick = message.time
                open_notes = last_note_on[message.note]

                notes_to_close = [
                    start_tick
                    for start_tick in open_notes
                    if start_tick != end_tick
                ]
                notes_to_keep = [
                    start_tick
                    for start_tick in open_notes
                    if start_tick == end_tick
                ]

                for start_tick in notes_to_close:
                    messages.append(
                        {
                            "type": "note",
                            "data": {
                                "pitch": message.note,
                                "start": start_tick,
                                "end": end_tick,
                                "velocity": message.velocity,
                            },
                            "tick": start_tick,
                            "channel": message.channel,
                        }
                    )

                if len(notes_to_close) > 0 and len(notes_to_keep) > 0:
                    # Note-on on the same tick but we already closed
                    # some previous notes -> it will continue, keep it.
                    last_note_on[message.note] = notes_to_keep
                else:
                    # Remove the last note on for this instrument
                    del last_note_on[message.note]

    return messages, meta_messages


def extract_midi_data(mid: mido.MidiFile = None, load_path: str = None):
    """Returns MIDI data in an intermediate form for tokenization.

    Args:
        mid (mido.MidiFile, optional): MIDI to parse. Defaults to None.
        load_path (str, optional): Optionally, specify path to load mid from.
            to None.

    Returns:
        tuple[list, list]: Extracted messages and meta_messages.
    """
    if mid is None and load_path is None:
        raise ValueError
    elif mid is None and load_path is not None:
        mid = mido.MidiFile(load_path)

    # Convert time in mid to absolute
    for track in mid.tracks:
        curr_tick = 0
        for message in track:
            message.time += curr_tick
            curr_tick = message.time

    messages = []
    meta_messages = []
    for mid_track in mid.tracks:
        track_messages, track_meta_messages = _extract_track_data(mid_track)
        messages += track_messages
        meta_messages += track_meta_messages

    messages = sorted(messages, key=lambda x: x["tick"])

    return messages, meta_messages
