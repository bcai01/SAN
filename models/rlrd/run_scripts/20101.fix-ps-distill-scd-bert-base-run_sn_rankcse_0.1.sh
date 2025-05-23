#!/bin/bash
set -eu

SEED=61507

output_dir=${BC_OUTPUT_BASE_DIR_FOR_RMTK}/20101.fix-ps-distill-scd-bert-base-run_sn_rankcse_0.1

python train_1.py \
    --baseE_sim_thresh_upp 0.9999 \
    --baseE_sim_thresh_low 0.5 \
    --baseE_lmb 0.05 \
    --t_lmb 0.001 \
    --simf Spearmanr \
    --loss_type weighted_sum \
    --corpus_vecs $BC_MODEL_PATH_FOR_RMTK/rankcse/index_vecs_rank1/corpus_0.01_sncse.npy \
    --model_name_or_path $BC_BERT_BASE_PATH \
    --train_file ${BC_OUTPUT_BASE_DIR_FOR_CORPUS}/001.from-prev-exp/wiki11m_soft_neg_rr_mask_1-4_tsv.csv \
    --num_train_epochs 4 \
    --per_device_train_batch_size 128 \
    --learning_rate 3e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --metric_for_best_model stsb_spearman \
    --eval_step 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16 \
    --first_teacher_name_or_path $BC_PERTRAINED_MODEL_BASE_DIR/wxt-SNCSE-rank/ \
    --second_teacher_name_or_path $BC_PERTRAINED_MODEL_BASE_DIR/wxt-simcse_bert_large/ \
    --distillation_loss listmle \
    --alpha_ 0.50 \
    --beta_ 1.0 \
    --gamma_ 1.0 \
    --tau2 0.05 \
    --soft_negative_file $BC_SNS_PATH \
    --output_dir $output_dir \
    --save_steps 125 \
    --sent0_cname text \
    --sent1_cname text \
    --sent2_cname re_mask4 \
    --scd_temp 0.1 \
    --enable_custom_dropout_for_last_column \
    --dropout_for_last_column 0.2 \
    --num_columns 4 \
    --use_ps_dist_loss \
    --ps_dist_loss_weight 0.2 \
    --z1_z4_tau2 0.05 \
    --enable_fix_sd_mlm_defects_using_sns \
    # --dropout 0.2 \
    


python evaluation_rank.py \
    --model_name_or_path $output_dir \
    --task_set sts \
    --mode test | tee -a $output_dir/eval_results.txt