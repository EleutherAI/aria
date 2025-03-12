python /home/loubb/work/aria/paper/scripts/build_embedding_eval_datasets.py \
    --model m3 \
    --model_cp_path /home/loubb/work/clamp3/weights_clamp3_saas_h_size_768_t_model_FacebookAI_xlm-roberta-base_t_length_128_a_size_768_a_layers_12_a_length_128_s_size_768_s_layers_12_p_size_64_p_length_512.pth \
    --dataset_load_path /mnt/ssd1/aria/data/paper/clas/pianist/train-mididict.jsonl \
    --dataset_save_path /mnt/ssd1/aria/data/paper/clas/pianist/train-clamp.jsonl \
    --compute_per_file_embeddings

python /home/loubb/work/aria/paper/scripts/build_embedding_eval_datasets.py \
    --model m3 \
    --model_cp_path /home/loubb/work/clamp3/weights_clamp3_saas_h_size_768_t_model_FacebookAI_xlm-roberta-base_t_length_128_a_size_768_a_layers_12_a_length_128_s_size_768_s_layers_12_p_size_64_p_length_512.pth \
    --dataset_load_path /mnt/ssd1/aria/data/paper/clas/pianist/test-mididict.jsonl \
    --dataset_save_path /mnt/ssd1/aria/data/paper/clas/pianist/test-clamp.jsonl \
    --compute_per_file_embeddings
