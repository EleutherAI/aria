python /home/loubb/work/aria/paper/scripts/build_embedding_eval_datasets.py \
    --model m3 \
    --model_cp_path /home/loubb/work/clamp3/weights_m3_p_size_64_p_length_512_t_layers_3_p_layers_12_h_size_768_lr_0.0001_batch_16_mask_0.45.pth \
    --dataset_load_path /mnt/ssd1/aria/data/paper/clas/pianist/train-mididict.jsonl \
    --dataset_save_path /mnt/ssd1/aria/data/paper/clas/pianist/train-m3.jsonl \
    --compute_per_file_embeddings \
    --m3_is_encoder_checkpoint

python /home/loubb/work/aria/paper/scripts/build_embedding_eval_datasets.py \
    --model m3 \
    --model_cp_path /home/loubb/work/clamp3/weights_m3_p_size_64_p_length_512_t_layers_3_p_layers_12_h_size_768_lr_0.0001_batch_16_mask_0.45.pth \
    --dataset_load_path /mnt/ssd1/aria/data/paper/clas/pianist/test-mididict.jsonl \
    --dataset_save_path /mnt/ssd1/aria/data/paper/clas/pianist/test-m3.jsonl \
    --compute_per_file_embeddings \
    --m3_is_encoder_checkpoint
