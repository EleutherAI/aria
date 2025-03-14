python /home/loubb/work/aria/paper/scripts/build_embedding_eval_datasets.py \
    --model aria \
    --model_cp_path /home/loubb/work/aria/models/emb-t0.1-s2048-e25.safetensors \
    --dataset_load_path /mnt/ssd1/aria/data/mididict-ft_val.jsonl \
    --dataset_save_path /mnt/ssd1/aria/data/finetune/ft-val_emb.jsonl \
    --compute_per_file_embeddings \
    --aria_max_batch_size 128

aria pretrain-dataset \
    -tokenizer_name abs \
    -load_path /mnt/ssd1/aria/data/mididict-ft_val.jsonl \
    -embedding_dataset_path /mnt/ssd1/aria/data/finetune/ft-val_emb.jsonl \
    -save_dir /mnt/ssd1/aria/data/finetune/val \
    -l 8192 \
    -e 1 \
    -sep_sequences