"""Utils for data/MIDI processing."""

import mido

from collections import defaultdict
from copy import deepcopy


# TODO:
# - Possibly refactor names 'mid' to 'midi' - DO THIS
# - Write proper tests


class MidiDict:
    """Container for MIDI data in dictionary form.

    The arguments must contain messages of the following format:

    meta_msg:
    {
        "type": "text" or "copyright",
        "data": str,
    }

    tempo_msg:
    {
        "type": "tempo",
        "data": int,
        "tick": int,
    }

    pedal_msg:
    {
        "type": "pedal",
        "data": 1 if pedal on; 0 if pedal off,
        "tick": int,
        "channel": int,
    }

    instrument_msg:
    {
        "type": "instrument",
        "data": int,
        "tick": int,
        "channel": int,
    }

    note_msg:
    {
        "type": "note",
        "data": {
            "pitch": int,
            "start": tick of note_on message,
            "end": tick of note_off message,
            "velocity": int,
        },
        "tick": tick of note_on message,
        "channel": int
    }

    Args:
        meta_msgs (list): Text or copyright MIDI meta messages.
        tempo_msgs (list): Tempo messages corresponding to the set_tempo MIDI
            message.
        pedal_msgs (list): Pedal on/off messages corresponding to control
            change 64 MIDI messages.
        instrument_msgs (list): Instrument messages corresponding to program
            change MIDI messages
        note_msgs (list): Note messages corresponding to matching note-on and
            note-off MIDI messages.
    """

    def __init__(
        self,
        meta_msgs: list,
        tempo_msgs: list,
        pedal_msgs: list,
        instrument_msgs: list,
        note_msgs: list,
        ticks_per_beat: int,
    ):
        self.meta_msgs = meta_msgs
        self.tempo_msgs = tempo_msgs
        self.pedal_msgs = pedal_msgs
        self.instrument_msgs = instrument_msgs
        self.note_msgs = note_msgs
        self.ticks_per_beat = ticks_per_beat  # Possibly refactor

        # This combines the individual dictionaries into one
        self.program_to_instrument = (
            {i: "piano" for i in range(0, 7 + 1)}
            | {i: "chromatic" for i in range(8, 15 + 1)}
            | {i: "organ" for i in range(16, 23 + 1)}
            | {i: "guitar" for i in range(24, 31 + 1)}
            | {i: "bass" for i in range(32, 39 + 1)}
            | {i: "strings" for i in range(40, 47 + 1)}
            | {i: "ensemble" for i in range(48, 55 + 1)}
            | {i: "brass" for i in range(56, 63 + 1)}
            | {i: "reed" for i in range(64, 71 + 1)}
            | {i: "pipe" for i in range(72, 79 + 1)}
            | {i: "synth_lead" for i in range(80, 87 + 1)}
            | {i: "synth_pad" for i in range(88, 95 + 1)}
            | {i: "synth_effect" for i in range(96, 103 + 1)}
            | {i: "ethnic" for i in range(104, 111 + 1)}
            | {i: "percussive" for i in range(112, 119 + 1)}
            | {i: "sfx" for i in range(120, 127 + 1)}
        )

    def get_msg_dict(self):
        return {
            "meta_msgs": self.meta_msgs,
            "tempo_msgs": self.tempo_msgs,
            "pedal_msgs": self.pedal_msgs,
            "instrument_msgs": self.instrument_msgs,
            "note_msgs": self.note_msgs,
            "ticks_per_beat": self.ticks_per_beat,
        }

    def to_midi(self):
        """Inplace version of dict_to_midi."""
        return dict_to_midi(self.get_msg_dict())

    @classmethod
    def from_msg_dict(cls, msg_dict: dict):
        """Inplace version of midi_to_dict."""
        assert msg_dict.keys() == {
            "meta_msgs",
            "tempo_msgs",
            "pedal_msgs",
            "instrument_msgs",
            "note_msgs",
            "ticks_per_beat",
        }

        return cls(**msg_dict)

    @classmethod
    def from_midi(cls, mid: mido.MidiFile):
        """Inplace version of midi_to_dict."""
        return cls(**midi_to_dict(mid))

    # TODO:
    # - Add remove drums (aka remove channel 9&16) pre-processing
    # - Add similar method for removing specific programs
    # - Decide whether this is necessary to have here in pre-precessing
    def remove_instruments(self, config: dict):
        """Removes all messages with instruments specified in config, excluding
        drums."""
        programs_to_remove = [
            i
            for i in range(1, 127 + 1)
            if config[self.program_to_instrument[i]] is True
        ]
        channels_to_remove = [
            msg["channel"]
            for msg in self.instrument_msgs
            if msg["data"] in programs_to_remove
        ]

        # Remove drums (channel 9/16) from channels to remove
        channels_to_remove = [i for i in channels_to_remove if i not in {9, 16}]

        # Remove unwanted messages all type by looping over msgs types. We need
        # to remove ticks_per_beat as this does not contain any messages.
        msg_dict = {
            k: v
            for k, v in self.get_msg_dict().items()
            if k != "ticks_per_beat"
        }
        for msgs_name, msgs_list in msg_dict.items():
            setattr(
                self,
                msgs_name,
                [
                    msg
                    for msg in msgs_list
                    if msg.get("channel", -1) not in channels_to_remove
                ],
            )


# Loosely inspired by pretty_midi
def _extract_track_data(track: mido.MidiTrack):
    meta_msgs = []
    tempo_msgs = []
    pedal_msgs = []
    instrument_msgs = []
    note_msgs = []

    last_note_on = defaultdict(list)
    for message in track:
        # Meta messages
        if message.is_meta is True:
            if message.type == "text" or message.type == "copyright":
                meta_msgs.append(
                    {
                        "type": message.type,
                        "data": message.text,
                    }
                )
            # Tempo messages
            elif message.type == "set_tempo":
                tempo_msgs.append(
                    {
                        "type": "tempo",
                        "data": message.tempo,
                        "tick": message.time,
                    }
                )
        # Instrument messages
        elif message.type == "program_change":
            instrument_msgs.append(
                {
                    "type": "instrument",
                    "data": message.program,
                    "tick": message.time,
                    "channel": message.channel,
                }
            )
        # Pedal messages
        elif message.type == "control_change" and message.control == 64:
            if message.value == 0:
                val = 0
            else:
                val = 1
            pedal_msgs.append(
                {
                    "type": "pedal",
                    "data": val,
                    "tick": message.time,
                    "channel": message.channel,
                }
            )
        # Note messages
        elif message.type == "note_on" and message.velocity > 0:
            last_note_on[message.note].append((message.time, message.velocity))
        elif message.type == "note_off" or (
            message.type == "note_on" and message.velocity == 0
        ):
            # Ignore non-existent note-ons
            if message.note in last_note_on:
                end_tick = message.time
                open_notes = last_note_on[message.note]

                notes_to_close = [
                    (start_tick, velocity)
                    for start_tick, velocity in open_notes
                    if start_tick != end_tick
                ]
                notes_to_keep = [
                    (start_tick, velocity)
                    for start_tick, velocity in open_notes
                    if start_tick == end_tick
                ]

                for start_tick, velocity in notes_to_close:
                    note_msgs.append(
                        {
                            "type": "note",
                            "data": {
                                "pitch": message.note,
                                "start": start_tick,
                                "end": end_tick,
                                "velocity": velocity,
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

    return {
        "meta_msgs": meta_msgs,
        "tempo_msgs": tempo_msgs,
        "pedal_msgs": pedal_msgs,
        "instrument_msgs": instrument_msgs,
        "note_msgs": note_msgs,
    }


# TODO: Redo docstring
def midi_to_dict(mid: mido.MidiFile):
    """Returns MIDI data in an intermediate dictionary form for tokenization.

    Args:
        mid (mido.MidiFile, optional): MIDI to parse.

    Returns:
        dict: Data extracted from the MIDI file. This dictionary has the
            entries "meta_msgs", "tempo_msgs", "pedal_msgs", "instrument_msgs",
            "note_msgs".
    """
    mid = deepcopy(mid)

    # Convert time in mid to absolute
    for track in mid.tracks:
        curr_tick = 0
        for message in track:
            message.time += curr_tick
            curr_tick = message.time

    # Compile track data
    data = defaultdict(list)
    for mid_track in mid.tracks:
        for k, v in _extract_track_data(mid_track).items():
            data[k] += v

    # Sort by tick
    for k, v in data.items():
        data[k] = sorted(v, key=lambda x: x.get("tick", 0))

    # Add ticks per beat
    data["ticks_per_beat"] = mid.ticks_per_beat

    return data


# TODO: Redo docstring
def dict_to_midi(mid_data: dict):
    """Converts MIDI information from dictionary form into a mido.MidiFile.

    This function performs midi_to_dict in reverse. For a complete
    description of the correct input format, see the attributes of the MidiDict
    class.

    Args:
        mid_data (dict): MIDI information in dictionary form. See midi_to_dict
            for a complete description.

    Returns:
        mido.MidiFile: The MIDI parsed from the input data.
    """
    mid_data = deepcopy(mid_data)

    assert set(mid_data.keys()) <= {
        "meta_msgs",
        "tempo_msgs",
        "pedal_msgs",
        "instrument_msgs",
        "note_msgs",
        "ticks_per_beat",
    }, "Invalid json/dict."

    ticks_per_beat = mid_data.pop("ticks_per_beat")
    if "meta_msgs" in mid_data.keys():
        del mid_data["meta_msgs"]

    # Add all messages (not ordered) to one track
    track = mido.MidiTrack()
    for msgs in mid_data.values():
        for msg in msgs:
            if msg["type"] == "tempo":
                track.append(
                    mido.MetaMessage(
                        "set_tempo", tempo=msg["data"], time=msg["tick"]
                    )
                )
            elif msg["type"] == "pedal":
                track.append(
                    mido.Message(
                        "control_change",
                        control=64,
                        value=msg["data"] * 127,  # Stored in dict as 1 or 0
                        channel=msg["channel"],
                        time=msg["tick"],
                    )
                )
            elif msg["type"] == "instrument":
                track.append(
                    mido.Message(
                        "program_change",
                        program=msg["data"],
                        channel=msg["channel"],
                        time=msg["tick"],
                    )
                )
            elif msg["type"] == "note":
                # Note on
                track.append(
                    mido.Message(
                        "note_on",
                        note=msg["data"]["pitch"],
                        velocity=msg["data"]["velocity"],
                        channel=msg["channel"],
                        time=msg["data"]["start"],
                    )
                )
                # Note off
                track.append(
                    mido.Message(
                        "note_on",
                        note=msg["data"]["pitch"],
                        velocity=0,
                        channel=msg["channel"],
                        time=msg["data"]["end"],
                    )
                )

    # Sort and convert from abs_time -> delta_time
    track = sorted(track, key=lambda msg: msg.time)
    tick = 0
    for msg in track:
        msg.time -= tick
        tick += msg.time

    track.append(mido.MetaMessage("end_of_track", time=0))
    mid = mido.MidiFile(type=0)
    mid.ticks_per_beat = ticks_per_beat
    mid.tracks.append(track)

    return mid


def _test_max_programs(midi_dict: MidiDict, max: int):
    """Returns false if midi_dict uses more than {max} programs."""
    present_programs = set(
        map(
            lambda msg: msg["data"],
            midi_dict.instrument_msgs,
        )
    )

    if len(present_programs) <= max:
        return True
    else:
        return False


def _test_max_instruments(midi_dict: MidiDict, max: int):
    present_instruments = set(
        map(
            lambda msg: midi_dict.program_to_instrument[msg["data"]],
            midi_dict.instrument_msgs,
        )
    )

    if len(present_instruments) <= max:
        return True
    else:
        return False


def _test_no_notes(midi_dict: MidiDict):
    if len(midi_dict.note_msgs) > 0:
        return True
    else:
        return False
