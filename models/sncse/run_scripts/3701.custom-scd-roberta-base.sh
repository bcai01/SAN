#!/bin/bash

set -eu


output_dir=${BC_OUTPUT_BASE_DIR_FOR_SNCSE}/3701.custom-scd-roberta-base

cd SNCSE
python train_SNCSE.py \
    --model_name_or_path $BC_ROBERTA_BASE_PATH \
    --train_file ${BC_OUTPUT_BASE_DIR_FOR_CORPUS}/001.from-prev-exp/wiki11m_soft_neg_rr_mask_1-4_tsv.csv \
    --output_dir $output_dir  \
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
    --sent0_cname text \
    --sent1_cname text \
    --sent2_cname re_mask4 \
    --scd_temp 0.1 \
    --enable_custom_dropout_for_last_column \
    --dropout_for_last_column 0.2 \
    --num_columns 4 \
    
cp ../$0 $output_dir/run_unsup.sh

python evaluation.py \
    --model_name_or_path $output_dir \
    --task_set sts \
    --mode test | tee -a $output_dir/eval_results.txt