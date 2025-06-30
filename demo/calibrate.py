import argparse
import sys
import threading
import time

import mido

MIDDLE_C = 60
C_MAJOR_CHORD = [MIDDLE_C, 64, 67, 72]  # C4, E4, G4, C5


def schedule_note_off(port: mido.ports.BaseOutput, note: int, delay: float):
    """Schedules a non-blocking MIDI note-off message."""

    def _off():
        port.send(mido.Message("note_off", note=note, velocity=0))

    t = threading.Timer(delay, _off)
    t.daemon = True  # Allow main program to exit even if timers are pending
    t.start()


def strike(
    port: mido.ports.BaseOutput, velocity: int, offset_ms: int, notes: list[int]
):
    """
    Performs a "3-2-1-GO!" countdown, sending MIDI notes with a precise offset.
    The note-on message is sent `offset_ms` *before* "GO!" is printed.
    """
    offset_sec = offset_ms / 1000.0

    print("3")
    time.sleep(1)
    print("2")
    time.sleep(1)
    print("1")

    # Use monotonic time for a clock that is not affected by system time changes
    go_time = time.monotonic() + 1.0
    note_on_time = go_time - offset_sec

    # Wait until the calculated time to send the MIDI message
    sleep_duration = note_on_time - time.monotonic()
    if sleep_duration > 0:
        time.sleep(sleep_duration)

    for note in notes:
        port.send(mido.Message("note_on", note=note, velocity=velocity))
        schedule_note_off(port, note, delay=0.5)

    # Wait for the exact moment to print "GO!"
    sleep_duration = go_time - time.monotonic()
    if sleep_duration > 0:
        time.sleep(sleep_duration)

    print("GO!\n")


def note_repetition_trial(
    port: mido.ports.BaseOutput,
    velocity: int,
    notes: list[int],
    note_length_ms: int,
    gap_ms: int,
):
    """Plays a note or chord repeatedly for a 3-second trial period."""
    print("Playing 3-second loop...")

    note_length_sec = note_length_ms / 1000.0
    gap_sec = gap_ms / 1000.0
    end_time = time.monotonic() + 3.0

    while time.monotonic() < end_time:
        # Ensure there's enough time for one full note cycle before the end
        if time.monotonic() + note_length_sec + gap_sec > end_time:
            break

        for note in notes:
            port.send(mido.Message("note_on", note=note, velocity=velocity))

        time.sleep(note_length_sec)

        for note in notes:
            port.send(mido.Message("note_off", note=note, velocity=0))

        if gap_sec > 0:
            time.sleep(gap_sec)

    print("...loop finished.\n")


def calibrate_output_latency(
    port_name: str,
    velocity: int,
    step_ms: int,
    initial_offset_ms: int,
    chord_mode: bool,
):
    """Interactive loop to find the ideal hardware latency offset."""
    notes = C_MAJOR_CHORD if chord_mode else [MIDDLE_C]
    offset_ms = initial_offset_ms

    try:
        with mido.open_output(port_name) as port:
            print(f"Opened MIDI output: {port_name}\n")
            while True:
                strike(port, velocity, offset_ms, notes)
                print(f"Current offset: {offset_ms} ms")
                cmd = (
                    input("[u]p / [d]own / [r]epeat / [q]uit: ").strip().lower()
                )

                if cmd == "u":
                    offset_ms += step_ms
                elif cmd == "d":
                    offset_ms = max(0, offset_ms - step_ms)
                elif cmd == "q":
                    break
                # Any other key (incl. 'r' or enter) repeats the trial
                print()
    except (KeyboardInterrupt, SystemExit):
        print("\nInterrupted — exiting.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")


def calibrate_note_timing(
    port_name: str,
    velocity: int,
    step_ms: int,
    note_length_ms: int,
    initial_gap_ms: int,
    chord_mode: bool,
):
    """Interactive loop to find a comfortable note repetition speed."""
    notes = C_MAJOR_CHORD if chord_mode else [MIDDLE_C]
    gap_ms = initial_gap_ms

    try:
        with mido.open_output(port_name) as port:
            print(f"Opened MIDI output: {port_name}\n")
            while True:
                note_repetition_trial(
                    port, velocity, notes, note_length_ms, gap_ms
                )
                print(f"Current gap: {gap_ms} ms")
                cmd = (
                    input("[u]p / [d]own / [r]epeat / [q]uit: ").strip().lower()
                )

                if cmd == "u":
                    gap_ms += step_ms
                elif cmd == "d":
                    gap_ms = max(0, gap_ms - step_ms)
                elif cmd == "q":
                    break
                print()
    except (KeyboardInterrupt, SystemExit):
        print("\nInterrupted — exiting.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")


def measure_input_latency(port_name: str, timeout_sec: float = 2.0):
    """
    3-2-1-GO countdown → you strike a key on GO.
    Prints the latency (note_on arrival – GO).

    • Uses the same MIDI port for input.
    • Waits `timeout_sec` seconds for a note-on; repeats if none arrives.
    """
    try:
        with mido.open_ioport(port_name) as port:
            print(f"Opened MIDI I/O port: {port_name}\n")

            while True:
                # ── simple countdown ────────────────────────────────────
                for n in ("3", "2", "1"):
                    print(n)
                    time.sleep(1)

                go_time = time.monotonic()
                print("GO!")

                # wait for first note-on (velocity>0) or timeout
                deadline = go_time + timeout_sec
                latency_ms = None
                while time.monotonic() < deadline:
                    msg = port.poll()
                    if msg and msg.type == "note_on" and msg.velocity > 0:
                        latency_ms = (time.monotonic() - go_time) * 1000.0
                        break

                if latency_ms is None:
                    print("No key press detected – try again.\n")
                else:
                    print(f"Input latency: {latency_ms:.1f} ms\n")

                if input("[r]etry / [q]uit: ").strip().lower() == "q":
                    break
                print()

    except (KeyboardInterrupt, SystemExit):
        print("\nInterrupted — exiting.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")


def list_midi_ports() -> None:
    """Prints a list of available MIDI output ports."""
    print("Available MIDI output ports:")
    try:
        port_names = mido.get_output_names()
        if not port_names:
            print("  (No ports found)")
        for name in port_names:
            print(f"  - {name}")
    except Exception as e:
        print(f"Could not retrieve MIDI ports: {e}")


def parse_args():
    """Parses command-line arguments for the calibration tool."""
    parser = argparse.ArgumentParser(
        description="A tool to calibrate Disklavier latency and note timing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── global option ──────────────────────────────────────────────────────
    parser.add_argument(
        "--list-ports",
        action="store_true",
        help="List available MIDI output ports and exit.",
    )

    # ── options common to *all* modes ─────────────────────────────────────
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--port", "-p", required=True, help="MIDI port name.")
    parent.add_argument(
        "--velocity",
        "-v",
        type=int,
        default=80,
        help="Note-on velocity (1-127).",
    )
    parent.add_argument(
        "--step",
        "-s",
        type=int,
        default=10,
        help="Adjustment step in ms (latency/timing modes).",
    )
    parent.add_argument(
        "--chord",
        "-c",
        action="store_true",
        help="Use a C-major chord instead of single note.",
    )

    sub = parser.add_subparsers(dest="command", help="Available commands.")

    # ── output-latency calibration ────────────────────────────────────────
    p_lat = sub.add_parser(
        "output",
        parents=[parent],
        help="Calibrate output latency.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_lat.add_argument(
        "--offset",
        "-o",
        type=int,
        default=100,
        help="Initial latency offset in ms.",
    )

    # ── repeated-note timing calibration ──────────────────────────────────
    p_tim = sub.add_parser(
        "timing",
        parents=[parent],
        help="Calibrate minimum gap between notes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_tim.add_argument(
        "--note-length",
        "-l",
        type=int,
        default=500,
        help="Note duration in ms.",
    )
    p_tim.add_argument(
        "--gap",
        "-g",
        type=int,
        default=100,
        help="Initial gap between notes in ms.",
    )

    # ── input-latency measurement (new) ───────────────────────────────────
    p_in = sub.add_parser(
        "input",
        parents=[parent],
        help="Measure input latency (countdown → strike).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_in.add_argument(
        "--timeout",
        "-t",
        type=float,
        default=2.0,
        help="Seconds to wait for a key press before retry.",
    )

    args = parser.parse_args()

    # global flag handler
    if args.list_ports:
        list_midi_ports()
        sys.exit(0)

    if not args.command:
        parser.error(
            "A command is required: choose 'output', 'timing', or 'input'."
        )

    return args


def main():
    """Dispatches to the selected calibration or measurement routine."""
    args = parse_args()

    if args.command == "output":
        calibrate_output_latency(
            port_name=args.port,
            velocity=args.velocity,
            step_ms=args.step,
            initial_offset_ms=args.offset,
            chord_mode=args.chord,
        )

    elif args.command == "timing":
        calibrate_note_timing(
            port_name=args.port,
            velocity=args.velocity,
            step_ms=args.step,
            note_length_ms=args.note_length,
            initial_gap_ms=args.gap,
            chord_mode=args.chord,
        )

    elif args.command == "input":
        measure_input_latency(
            port_name=args.port,
            timeout_sec=args.timeout,
        )


if __name__ == "__main__":
    main()
