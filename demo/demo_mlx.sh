# # MID_PATH="/Users/louis/Library/CloudStorage/Dropbox/shared/audio_piano.mid"
# MID_PATH="/Users/louis/Library/CloudStorage/Dropbox/shared/bill_evans.mid"
#     # --midi_path ${MID_PATH} \
#     # --midi_through "IAC Driver Bus 3" \

# python /Users/louis/work/aria/demo/demo_mlx.py \
#     --checkpoint /Users/louis/work/aria/models/medium-75-ft.safetensors \
#     --embedding_checkpoint /Users/louis/work/aria/models/medium-emb.safetensors \
#     --embedding_midi /Users/louis/Library/CloudStorage/Dropbox/shared/prompt/noodle.mid \
#     --midi_in "Scarlett 18i8 USB" \
#     --midi_out "Scarlett 18i8 USB" \
#     --midi_control_signal 67 \
#     --midi_reset_control_signal 66 \
#     --save_path /Users/louis/Dropbox/shared/output.mid \
#     --quantize \
#     --temp 0.98 \
#     --min_p 0.035

####

MID_PATH="./example-prompts/nocturne.mid"

python ./demo/demo_mlx.py \
    --checkpoint ./models/medium-75-annealed.safetensors \
    --midi_path ${MID_PATH} \
    --midi_through "IAC Driver Bus 2" \
    --midi_out "IAC Driver Bus 3" \
    --save_path ./output.mid \
    --quantize \
    --temp 0.98 \
    --min_p 0.035
