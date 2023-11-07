"""Contains miscellaneous utilities"""

import os
import subprocess
import requests

from pydub import AudioSegment


def midi_to_audio(mid_path: str, soundfont_path: str | None = None):
    SOUNDFONT_PATH = "fluidsynth/DoreMarkYamahaS6-v1.6.sf2"
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
