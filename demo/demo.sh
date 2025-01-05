python /home/loubb/work/aria/demo/demo.py \
    -cp /mnt/ssd1/aria/v2/medium-75-ft.safetensors \
    -midi_path /home/loubb/Dropbox/shared/prompt/nocturne.mid \
    -midi_out "Midi Through:Midi Through Port-1" \
    -midi_through "Midi Through:Midi Through Port-2" \
    -save_path /home/loubb/Dropbox/shared/output.mid \
    -midi_control_signal 66 \
    -temp 0.95
