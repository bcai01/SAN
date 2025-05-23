#!/bin/bash

# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.

set -eu

output_dir=${BC_OUTPUT_BASE_DIR_FOR_SIMCSE}/1604.custom-model-scd-bert-large


python train.py \
    --model_name_or_path $BC_BERT_LARGE_PATH \
    --train_file ${BC_OUTPUT_BASE_DIR_FOR_CORPUS}/001.from-prev-exp/wiki11m_soft_neg_rr_mask_1-4_tsv.csv \
    --output_dir $output_dir \
    --num_train_epochs 1 \
    --per_device_train_batch_size 64 \
    --learning_rate 1e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --sent0_cname text \
    --sent1_cname text \
    --sent2_cname re_mask4 \
    --scd_temp 0.1 \
    --enable_custom_dropout_for_last_column \
    --dropout_for_last_column 0.2 \
    --num_columns 3 \

    
    # --dropout 0.15 \
    # --fp16 \
    # --dropout 0.2 \
    # "$@"

cp $0 $output_dir/run_unsup.sh

python evaluation.py \
    --model_name_or_path $output_dir \
    --pooler cls_before_pooler \
    --task_set sts \
    --mode test | tee -a $output_dir/eval_results.txt