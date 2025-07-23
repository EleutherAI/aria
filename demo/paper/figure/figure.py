# /// script
# requires-python = ">=3.8"
# dependencies = [
#   # ### --- SIMPLIFIED --- ###: Pillow is no longer needed
#   "matplotlib",
#   "ariautils @ https://github.com/EleutherAI/aria-utils.git"
# ]
# ///

import argparse
import math
from pathlib import Path
from typing import List, Literal, Optional, TypedDict

# ### --- SIMPLIFIED --- ###: Matplotlib is now the primary plotting library
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ariautils.midi import MidiDict

# --- Hardcoded Configuration Constants ---
# You can modify these values directly as needed.

# Note Pitch Range (inclusive)
MIN_PITCH = 52
MAX_PITCH = 80

# Image Dimensions and Resolution
DPI = 500
IMG_WIDTH_CM = 35
IMG_HEIGHT_CM = 4

# Visual Style
NOTE_SPACING_RATIO = (
    0.4  # 0.0 for no spacing, 0.15 means 15% of row height is spacing
)
LINE_THICKNESS_PX = 12
DASHED_LINE_PATTERN_PX = (61, 17)  # (dash_length, gap_length) in pixels

# Figure Mode Constants
CHUNK_LEN_MS = 690  # Length of a prefill chunk for 'prefill' mode

# Colors
COLOR_YELLOW_BG = "#FFF9C4"
COLOR_GRAY_LINE = "#A9A9A9"
COLOR_RED_BG = "#FFB3B3"
COLOR_BLUE_NOTE = "#4A90E2"
COLOR_GREEN_NOTE = "#4CAF50"

# --- End of Configuration ---


class NoteToDraw(TypedDict):
    """A simple structure to hold note information for drawing."""

    pitch: int
    start_ms: int
    end_ms: int


def create_pianoroll_pdf(
    midi_path: Path,
    output_path: Path,
    cutoff_ms: int,
    total_duration_ms: int,
    truncate: bool,
    mode: Literal["none", "prefill", "recalc", "generate"],
    generation_len_ms: Optional[int] = None,
):
    """
    Generates a minimalist, vector-based (PDF) piano roll from a MIDI file.
    """
    print(f"Loading MIDI file: {midi_path.name}")
    try:
        midi_data = MidiDict.from_midi(midi_path)
    except Exception as e:
        print(f"Error loading MIDI file with MidiDict: {e}")
        return

    # Determine the total time span to display notes for
    display_until_ms = total_duration_ms
    if mode == "generate" and generation_len_ms is not None:
        display_until_ms = cutoff_ms + generation_len_ms

    # 1. Filter and collect all potentially visible notes
    all_notes: List[NoteToDraw] = []
    time_offset_ms = -300
    for note_msg in midi_data.note_msgs:
        pitch = note_msg["data"]["pitch"]
        if not (MIN_PITCH <= pitch <= MAX_PITCH):
            continue
        start_ms = midi_data.tick_to_ms(note_msg["data"]["start"])
        if start_ms > display_until_ms:
            continue
        end_ms = midi_data.tick_to_ms(note_msg["data"]["end"])
        if mode == "none" and truncate and end_ms > cutoff_ms:
            end_ms = cutoff_ms
        all_notes.append(
            {
                "pitch": pitch,
                "start_ms": start_ms + time_offset_ms,
                "end_ms": end_ms + time_offset_ms,
            }
        )

    # Filter notes to draw based on the mode and time window
    notes_to_draw = []
    filter_end_time = display_until_ms
    for n in all_notes:
        if n["start_ms"] < filter_end_time:
            notes_to_draw.append(n)

    if mode == "prefill":
        for note in notes_to_draw:
            if note["end_ms"] > cutoff_ms:
                note["end_ms"] = cutoff_ms

    print(f"Found {len(notes_to_draw)} notes to draw for mode '{mode}'.")

    # --- SHARED CALCULATIONS ---
    img_width_px = int(IMG_WIDTH_CM / 2.54 * DPI)
    img_height_px = int(IMG_HEIGHT_CM / 2.54 * DPI)
    num_pitches = MAX_PITCH - MIN_PITCH + 1
    pitch_row_height = img_height_px / num_pitches
    y_spacing = pitch_row_height * NOTE_SPACING_RATIO / 2.0
    line_half_width = LINE_THICKNESS_PX / 2.0

    # Boundary calculations
    cutoff_px_center = (cutoff_ms / total_duration_ms) * img_width_px
    cutoff_boundary_px = cutoff_px_center + line_half_width

    recalc_boundary_start_ms = 0
    if mode in ["recalc", "generate"]:
        truncated_notes = [
            n for n in notes_to_draw if n["start_ms"] <= cutoff_ms < n["end_ms"]
        ]
        if truncated_notes:
            first_trunc_start_ms = min(n["start_ms"] for n in truncated_notes)
            recalc_boundary_start_ms = first_trunc_start_ms

    yellow_end_ms = 0
    if mode == "prefill":
        yellow_end_ms = math.floor(cutoff_ms / CHUNK_LEN_MS) * CHUNK_LEN_MS
    elif mode in ["recalc", "generate"]:
        yellow_end_ms = recalc_boundary_start_ms

    # ### --- SIMPLIFIED --- ###: This is now the only drawing implementation
    # --- MATPLOTLIB (VECTOR) IMPLEMENTATION ---
    fig_width_in = IMG_WIDTH_CM / 2.54
    fig_height_in = IMG_HEIGHT_CM / 2.54

    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in), dpi=DPI)
    ax.set_rasterization_zorder(0)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.axis("off")

    ax.set_xlim(0, img_width_px)
    ax.set_ylim(0, img_height_px)
    ax.invert_yaxis()
    ax.set_facecolor("white")

    # 1. Backgrounds
    if yellow_end_ms > 0:
        x_boundary_yellow = (yellow_end_ms / total_duration_ms) * img_width_px
        ax.add_patch(
            patches.Rectangle(
                (0, 0),
                x_boundary_yellow,
                img_height_px,
                facecolor=COLOR_YELLOW_BG,
                zorder=1,
            )
        )

    if mode in ["recalc", "generate"] and recalc_boundary_start_ms < cutoff_ms:
        x_start_red = (
            recalc_boundary_start_ms / total_duration_ms
        ) * img_width_px
        ax.add_patch(
            patches.Rectangle(
                (x_start_red, 0),
                cutoff_boundary_px - x_start_red,
                img_height_px,
                facecolor=COLOR_RED_BG,
                zorder=1,
            )
        )

    # 2. Notes
    for note in notes_to_draw:
        note_end_ms = min(note["end_ms"], display_until_ms)
        x0 = (note["start_ms"] / total_duration_ms) * img_width_px
        x1 = (note_end_ms / total_duration_ms) * img_width_px
        if x1 <= x0:
            continue

        pitch_index = MAX_PITCH - note["pitch"]
        y0 = pitch_index * pitch_row_height + y_spacing
        height = pitch_row_height - 2 * y_spacing

        is_split_note = (
            mode in ["recalc", "generate"]
            and note["start_ms"] <= cutoff_ms < note["end_ms"]
        )
        is_new_note = mode == "generate" and note["start_ms"] >= cutoff_ms

        if is_split_note:
            ax.add_patch(
                patches.Rectangle(
                    (x0, y0),
                    cutoff_boundary_px - x0,
                    height,
                    facecolor="black",
                    zorder=2,
                )
            )
            ax.add_patch(
                patches.Rectangle(
                    (cutoff_boundary_px, y0),
                    x1 - cutoff_boundary_px,
                    height,
                    facecolor=COLOR_BLUE_NOTE,
                    zorder=2,
                )
            )
        elif is_new_note:
            ax.add_patch(
                patches.Rectangle(
                    (x0, y0),
                    x1 - x0,
                    height,
                    facecolor=COLOR_GREEN_NOTE,
                    zorder=2,
                )
            )
        else:
            ax.add_patch(
                patches.Rectangle(
                    (x0, y0), x1 - x0, height, facecolor="black", zorder=2
                )
            )

    # 3. Lines
    if yellow_end_ms > 0:
        for t in range(CHUNK_LEN_MS, int(yellow_end_ms) + 1, CHUNK_LEN_MS):
            x_chunk_center = (t / total_duration_ms) * img_width_px
            ax.add_patch(
                patches.Rectangle(
                    (x_chunk_center - line_half_width, 0),
                    LINE_THICKNESS_PX,
                    img_height_px,
                    facecolor=COLOR_GRAY_LINE,
                    zorder=3,
                )
            )

    # Convert pixel-based line styles to Matplotlib's point-based system
    lw_pt = LINE_THICKNESS_PX * 72 / DPI
    dash_pt = [d * 72 / DPI for d in DASHED_LINE_PATTERN_PX]
    ax.axvline(
        x=cutoff_px_center,
        color=COLOR_GRAY_LINE,
        linestyle=(0, dash_pt),
        linewidth=lw_pt,
        zorder=4,
    )

    # Add border
    ax.add_patch(
        patches.Rectangle(
            (0, 0),
            img_width_px,
            img_height_px,
            facecolor="none",
            edgecolor="black",
            linewidth=lw_pt,
            zorder=5,
        )
    )

    plt.savefig(output_path, format="pdf", bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    print(f"Successfully generated PDF asset: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a minimalist vector (PDF) piano roll from a MIDI file.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "midi_path", type=Path, help="Path to the input MIDI file."
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Path to save the output PDF image.",
    )
    parser.add_argument(
        "--cutoff-ms",
        type=int,
        required=True,
        help="Time in ms to capture notes up to. This is the main point of interest.",
    )
    parser.add_argument(
        "--total-duration-ms",
        type=int,
        required=True,
        help="Total time duration the piano roll's width should represent.",
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="If set, notes that end after the cutoff time will be visually cut off (only in 'none' mode).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="none",
        choices=["none", "prefill", "recalc", "generate"],
        help="Selects the visualization mode for the figure.",
    )
    parser.add_argument(
        "--generation-len-ms",
        type=int,
        help="Required for 'generate' mode. How many ms of notes to show after the cutoff.",
    )

    args = parser.parse_args()

    if args.output_path.suffix.lower() != ".pdf":
        args.output_path = args.output_path.with_suffix(".pdf")

    if args.mode == "generate" and args.generation_len_ms is None:
        parser.error(
            "--generation-len-ms is required when using --mode generate"
        )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    create_pianoroll_pdf(**vars(args))


if __name__ == "__main__":
    main()
