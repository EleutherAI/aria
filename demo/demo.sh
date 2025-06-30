MID_PATH="/home/loubb/Dropbox/shared/demo.mid"

python /home/loubb/work/aria/demo/demo.py \
    -cp /mnt/ssd1/aria/v2/medium-dedupe-pt-cont2/checkpoints/epoch18_step0/model.safetensors \
    -midi_path ${MID_PATH} \
    -midi_out "Midi Through:Midi Through Port-1" \
    -midi_through "Midi Through:Midi Through Port-2" \
    -save_path /home/loubb/Dropbox/shared/output.mid \
    -midi_control_signal 66 \
    -midi_end_signal 67 \
    -temp 0.98 \
    -min_p 0.02