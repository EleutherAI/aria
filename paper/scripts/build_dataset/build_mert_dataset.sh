python /home/loubb/work/aria/paper/scripts/build_embedding_eval_datasets.py \
    --model mert \
    --dataset_load_path /mnt/ssd1/aria/data/paper/clas/pianist/train-mididict.jsonl \
    --dataset_save_path /mnt/ssd1/aria/data/paper/clas/pianist/train-mert.jsonl \
    --mert_pianoteq_exec_path "/home/loubb/pianoteq/x86-64bit/Pianoteq 8 STAGE" \
    --mert_pianoteq_num_procs 16 \
    --compute_per_file_embeddings

python /home/loubb/work/aria/paper/scripts/build_embedding_eval_datasets.py \
    --model mert \
    --dataset_load_path /mnt/ssd1/aria/data/paper/clas/pianist/test-mididict.jsonl \
    --dataset_save_path /mnt/ssd1/aria/data/paper/clas/pianist/test-mert.jsonl \
    --mert_pianoteq_exec_path "/home/loubb/pianoteq/x86-64bit/Pianoteq 8 STAGE" \
    --mert_pianoteq_num_procs 16 \
    --compute_per_file_embeddings