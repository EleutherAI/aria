python /home/loubb/work/aria/paper/scripts/build_embedding_eval_datasets.py \
    --model mert \
    --dataset_load_path /mnt/ssd1/aria/data/paper/clas/genre-aria/train-mididict.jsonl \
    --dataset_save_path /mnt/ssd1/aria/data/paper/clas/genre-aria/train-mert.jsonl \
    --mert_pianoteq_exec_path "/home/loubb/pianoteq/x86-64bit/Pianoteq 8 STAGE" \
    --mert_pianoteq_num_procs 8