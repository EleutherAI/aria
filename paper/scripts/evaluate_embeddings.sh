MODEL="m3"
CATEGORY="pianist"
echo "Evaluating model ${MODEL} on category: ${CATEGORY}"

python /home/loubb/work/aria/paper/scripts/evaluate_embedding_with_probe.py \
    --model $MODEL \
    --metadata_category $CATEGORY \
    --train_dataset_path "/mnt/ssd1/aria/data/paper/clas/${CATEGORY}/train-${MODEL}.jsonl" \
    --test_dataset_path "/mnt/ssd1/aria/data/paper/clas/${CATEGORY}/test-${MODEL}.jsonl" \
    --num_epochs 50
