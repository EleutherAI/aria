python /home/loubb/work/aria/paper/scripts/build_embedding_eval_datasets.py \
    --model aria \
    --model_cp_path /home/loubb/work/aria/models/emb-t0.1-s2048-e25.safetensors \
    --dataset_load_path /mnt/ssd1/aria/data/paper/clas/pianist/train-mididict.jsonl \
    --dataset_save_path /mnt/ssd1/aria/data/paper/clas/pianist/train-aria.jsonl \
    --compute_per_file_embeddings \
    --aria_max_batch_size 128

python /home/loubb/work/aria/paper/scripts/build_embedding_eval_datasets.py \
    --model aria \
    --model_cp_path /home/loubb/work/aria/models/emb-t0.1-s2048-e25.safetensors \
    --dataset_load_path /mnt/ssd1/aria/data/paper/clas/pianist/test-mididict.jsonl \
    --dataset_save_path /mnt/ssd1/aria/data/paper/clas/pianist/test-aria.jsonl \
    --compute_per_file_embeddings \
    --aria_max_batch_size 128
