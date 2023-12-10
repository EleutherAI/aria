import os
import time

import fluidsynth
import mido
import requests
import torch
from huggingface_hub import snapshot_download
from rich.console import Console

from aria.config import load_model_config
from aria.model import ModelConfig, TransformerLM
from aria.sample import sample_top_p
from aria.tokenizer import TokenizerLazy

pprint = Console().print

from multiprocessing import Process, Queue

DOWNLOAD_URL = "https://www.dropbox.com/scl/fi/t8gou8stesm42sc559nzu/DoreMarkYamahaS6-v1.6.sf2?rlkey=28ecl63kkjjmwxrkd6hnzsq8f&dl=1"
SOUNDFONT_PATH = "fluidsynth/DoreMarkYamahaS6-v1.6.sf2"
USER_TIMEOUT = 2
MODEL_PATH = "reciprocate/aria-2711-125000"

def play(input_queue, output_queue):
    """Receive tokens and play them with fluidsynth"""

    # download soundfont if it's not already there
    if not os.path.isfile("fluidsynth/DoreMarkYamahaS6-v1.6.sf2"):
        if not os.path.isdir("fluidsynth"):
            os.mkdir("fluidsynth")
        print("Downloading soundfont ...")
        res = requests.get(url=DOWNLOAD_URL)
        if res.status_code == 200:
            with open(SOUNDFONT_PATH, "wb") as file_handle:
                file_handle.write(res.content)
            print("Download complete")
        else:
            print(f"Failed to download soundfont: RESPONSE {res.status_code}")
            return

    fs = fluidsynth.Synth(samplerate=44100.0, gain=1.0)
    fs.start(driver='coreaudio')

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

            if m == '<E>':
                finish = True
            elif m[0] == 'piano':
                fs.noteon(0, m[1], m[2])
                current_note = m[1]
            elif m[0] == 'dur':
                if current_note is not None:
                    open_notes[current_note] = m[1]
                    current_note = None
            elif m[0] == 'wait':
                time.sleep(m[1] / 1000)

                for note in list(open_notes.keys()):
                    open_notes[note] -= m[1]
                    if open_notes[note] <= 0:
                        del open_notes[note]
                        fs.noteoff(0, note)

if __name__ == '__main__':
    output_queue = Queue()
    input_queue = Queue()
    player = Process(target=play, args=(output_queue, input_queue))
    player.start()

    tokenizer = TokenizerLazy(return_tensors=True)
    device = torch.device('cpu')
    model_name = "large"
    model_config = ModelConfig(**load_model_config(model_name))
    model_config.set_vocab_size(tokenizer.vocab_size)
    model = TransformerLM(model_config).to(device)
    
    print(f"Loading {MODEL_PATH} ...")
    model_filepath = snapshot_download(MODEL_PATH) + "/model.safetensors"
    model_state = torch.load(model_filepath)
    model.load_state_dict(model_state)
    
    while mido.get_input_names() == []:
        print("Waiting for MIDI input device to connect ...")
        time.sleep(5)
    
    port_name = mido.get_input_names()[0]

    tokens = [
        ('prefix', 'instrument', 'piano'),
        '<S>',
    ]
    current_note = None

    while True:
        USER_TIMEOUT = 2
        is_ready = input_queue.get()

        pprint('>>>', style="magenta")
        input = mido.open_input(port_name)
        time_since = 0
        while True:
            m = input.poll()
            if m is None:
                time.sleep(0.0001)
                time_since += 0.0001
                if current_note and time_since > USER_TIMEOUT:
                    break
            elif not m.is_meta:
                if m.type == "note_on":
                    if current_note:
                        quantized_time = tokenizer._quantize_time(time_since * 1000)
                        tokens.append(('dur', quantized_time))
                        tokens.append(('wait', quantized_time))

                    # i found results to be better when i quantized the velocity like this
                    token = ('piano', m.note, TokenizerLazy._find_closest_int(m.velocity, [45, 60, 75, 90]))
                    tokens.append(token)
                    output_queue.put_nowait(token)
                    current_note = m.note
                    time_since = 0
                elif m.type == "note_off":
                    time_since = 0
                
        current_note = None
        pprint("<<<", style='magenta')
        past_kv = None
        temperature = 0.8
        
        input_ids = tokenizer.encode(tokens).unsqueeze(0).to(device)
    
        max_new_tokens = 32
        for _ in range(max_new_tokens):
            stime = time.time()
            logits, past_kv = model.forward(input_ids, use_cache=True, past_kv=past_kv)
            etime = time.time()
            logits = logits[:, -1, :]
          
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token_id = sample_top_p(probs, 0.9).view(-1)
            next_token = tokenizer.decode(next_token_id)

            output_queue.put_nowait(next_token[0])
            tokens.append(next_token[0])

            input_ids = next_token_id.unsqueeze(0)
        output_queue.put_nowait('<E>')
