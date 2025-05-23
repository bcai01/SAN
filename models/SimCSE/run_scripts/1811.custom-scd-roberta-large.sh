#!/bin/bash

# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.

set -eu

output_dir=${BC_OUTPUT_BASE_DIR_FOR_SIMCSE}/1811.custom-scd-roberta-large


python train.py \
    --model_name_or_path $BC_ROBERTA_LARGE_PATH \
    --train_file ${BC_OUTPUT_BASE_DIR_FOR_CORPUS}/001.from-prev-exp/wiki11m_soft_neg_rr_mask_1-4_tsv.csv \
    --output_dir $output_dir \
    --num_train_epochs 1 \
    --per_device_train_batch_size 256 \
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
    --fp16 \
    --sent0_cname text \
    --sent1_cname text \
    --sent2_cname re_mask4 \
    --scd_temp 0.5 \
    --enable_custom_dropout_for_last_column \
    --dropout_for_last_column 0.25 \
    --num_columns 3 \
    --dropout 0.15 \

cp $0 $output_dir/run_unsup.sh

python evaluation.py \
    --model_name_or_path $output_dir \
    --pooler cls_before_pooler \
    --task_set sts \
    --mode test | tee -a $output_dir/eval_results.txt