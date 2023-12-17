"""Contains miscellaneous utilities"""

import os
import subprocess
import time

import requests
from multiprocessing import Queue

from pydub import AudioSegment


def midi_to_audio(mid_path: str, soundfont_path: str | None = None):
    SOUNDFONT_PATH = os.path.join(
        os.path.dirname(__file__),
        "..",
        "fluidsynth/DoreMarkYamahaS6-v1.6.sf2",
    )
    DOWNLOAD_URL = "https://www.dropbox.com/scl/fi/t8gou8stesm42sc559nzu/DoreMarkYamahaS6-v1.6.sf2?rlkey=28ecl63kkjjmwxrkd6hnzsq8f&dl=1"

    if os.name != "posix":
        print("Conversion to mp3 is not supported for non-posix machines")
        return

    if not os.path.isfile(SOUNDFONT_PATH) and soundfont_path is None:
        _input = input(
            "fluidsynth soundfont missing, type Y to download and continue: "
        )
        if _input == "Y":
            if not os.path.isdir("fluidsynth"):
                os.mkdir("fluidsynth")

            res = requests.get(url=DOWNLOAD_URL)
            if res.status_code == 200:
                with open(SOUNDFONT_PATH, "wb") as file_handle:
                    file_handle.write(res.content)
                print("Download complete")
            else:
                print(f"Failed to download patch: RESPONSE {res.status_code}")
                return
        else:
            print("Aborting mp3 conversion")
            return

    if soundfont_path is None:
        soundfont_path = SOUNDFONT_PATH

    if mid_path.endswith(".mid") or mid_path.endswith(".midi"):
        base_path, _ = os.path.splitext(mid_path)
        wav_path = base_path + ".wav"
        mp3_path = base_path + ".mp3"

    try:
        process = subprocess.Popen(
            f"fluidsynth {soundfont_path} -g 0.7 --quiet --no-shell {mid_path} -T wav -F {wav_path}",
            shell=True,
        )
        process.wait()
        AudioSegment.from_wav(wav_path).export(mp3_path, format="mp3")
    except Exception as e:
        print("Failed to convert to mp3 using fluidsynth:")
        print(e)

    print(f"Saved files: \n{wav_path}\n{mp3_path}")


def _get_soundfont(path: str) -> bool:
    DOWNLOAD_URL = "https://www.dropbox.com/scl/fi/t8gou8stesm42sc559nzu/DoreMarkYamahaS6-v1.6.sf2?rlkey=28ecl63kkjjmwxrkd6hnzsq8f&dl=1"
    # download soundfont if it's not already there
    if not os.path.isfile(path):
        if not os.path.isdir("fluidsynth"):
            os.mkdir("fluidsynth")
        print("Downloading soundfont ...")
        res = requests.get(url=DOWNLOAD_URL)
        if res.status_code == 200:
            with open(path, "wb") as file_handle:
                file_handle.write(res.content)
            print("Download complete")
        else:
            print(f"Failed to download soundfont: RESPONSE {res.status_code}")
            return False
    return True


def _play(input_queue: Queue, output_queue: Queue):
    """
    Run in a separate process and receive tokens and play them with fluidsynth
    Credits to @maxreciprocate
    """
    SOUNDFONT_PATH = "fluidsynth/DoreMarkYamahaS6-v1.6.sf2"

    if not _get_soundfont(SOUNDFONT_PATH):
        return

    import fluidsynth  # lazy import
    import platform

    fs = fluidsynth.Synth()
    if platform.system() == "Linux":
        fs.start(driver="pulseaudio")
    else:
        fs.start()

    sfid = fs.sfload(SOUNDFONT_PATH)

    fs.program_select(0, sfid, 0, 0)

    output_queue.put_nowait(True)

    finish = False
    current_note = None
    open_notes = {}
    while True:
        if finish and input_queue.empty():
            output_queue.put_nowait(finish)
            finish = False
        elif not input_queue.empty():
            m = input_queue.get()
            print(m)
            if m is None:  # exit
                break
            elif m == "<E>":
                finish = True
            elif m[0] == "piano":
                fs.noteon(0, m[1], m[2])
                current_note = m[1]
            elif m[0] == "dur":
                if current_note is not None:
                    open_notes[current_note] = m[1]
                    current_note = None
            elif m[0] == "wait":
                time.sleep(m[1] / 1000)

                for note in list(open_notes.keys()):
                    open_notes[note] -= m[1]
                    if open_notes[note] <= 0:
                        del open_notes[note]
                        fs.noteoff(0, note)
        else:
            time.sleep(0.1)
