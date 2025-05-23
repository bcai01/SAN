#!/bin/bash
set -eu

output_dir=$BC_OUTPUT_BASE_DIR_FOR_RANKCSE/4601.custom-scd-bert-large

python train.py \
    --model_name_or_path $BC_BERT_LARGE_PATH \
    --train_file ${BC_OUTPUT_BASE_DIR_FOR_CORPUS}/001.from-prev-exp/wiki11m_soft_neg_rr_mask_1-4_tsv.csv \
    --output_dir $output_dir \
    --num_train_epochs 1 \
    --per_device_train_batch_size 128 \
    --learning_rate 2e-5 \
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
    --first_teacher_name_or_path $BC_DIFFCSE_BERT_BASE_PATH \
    --second_teacher_name_or_path $BC_OUTPUT_BASE_DIR_FOR_SIMCSE/1604.custom-model-scd-bert-large \
    --distillation_loss listmle \
    --alpha_ 0.33 \
    --beta_ 1.0 \
    --gamma_ 1.0 \
    --tau2 0.05 \
    --sent0_cname text \
    --sent1_cname text \
    --sent2_cname re_mask4 \
    --scd_temp 0.05 \
    --enable_custom_dropout_for_last_column \
    --dropout_for_last_column 0.2 \
    --num_columns 3 \
    --use_ps_dist_loss \
    --ps_dist_loss_weight 0.4 \
    --z1_z3_tau2 0.05 \

cp $0 $output_dir/run_unsup.sh

python evaluation.py \
    --model_name_or_path $output_dir \
    --pooler cls_before_pooler \
    --task_set sts \
    --mode test | tee -a $output_dir/eval_results.txt