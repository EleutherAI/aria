"""Includes tests for filtering MidiDict objects."""


def max_programs(mid_dict, **config):
    present_programs = []
    for msg in mid_dict.instrument_msgs:
        msg_program = msg["data"]
        if msg_program not in present_programs:
            present_programs.append(msg_program)

    if len(present_programs) <= config["max"]:
        return True
    else:
        return False


def max_instruments(mid_dict, **config):
    present_instruments = []
    for msg in mid_dict.instrument_msgs:
        msg_instrument = mid_dict.program_to_instrument[msg["data"]]
        if msg_instrument not in present_instruments:
            present_instruments.append(msg_instrument)

    if len(present_instruments) <= config["max"]:
        return True
    else:
        return False
