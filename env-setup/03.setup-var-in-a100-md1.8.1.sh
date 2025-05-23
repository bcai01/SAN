#!/bin/bash

export BC_PROJECT_BASE_DIR=../

export PYTHONPATH=${BC_PROJECT_BASE_DIR}:$PYTHONPATH

export BC_BERT_BASE_PATH=bert-base-uncased
export BC_BERT_LARGE_PATH=bert-large-uncased
export BC_ROBERTA_BASE_PATH=roberta-base
export BC_ROBERTA_LARGE_PATH=roberta-large

export BC_SIMCSE_BERT_LARGE_PATH=princeton-nlp/unsup-simcse-bert-large-uncased
export BC_DIFFCSE_BERT_BASE_PATH=voidism/diffcse-bert-base-uncased-sts
export BC_SIMCSE_ROBERTA_LARGE_PATH=princeton-nlp/unsup-simcse-roberta-large



export BC_WIKI1M_PATH=${BC_PROJECT_BASE_DIR}/corpus/wiki1m_for_simcse.txt
export BC_SNS_PATH=${BC_PROJECT_BASE_DIR}/corpus/soft_negative_samples.txt


export BC_OUTPUT_BASE_DIR_FOR_SIMCSE=${BC_PROJECT_BASE_DIR}/outputs/simcse
export BC_OUTPUT_BASE_DIR_FOR_RMTK=${BC_PROJECT_BASE_DIR}/outputs/rlrd
export BC_OUTPUT_BASE_DIR_FOR_SNCSE=${BC_PROJECT_BASE_DIR}/outputs/sncse
export BC_OUTPUT_BASE_DIR_FOR_RANKCSE=${BC_PROJECT_BASE_DIR}/outputs/rankcse
export BC_OUTPUT_RANKENCODER_CORPUS_10000_PATH=${BC_PROJECT_BASE_DIR}/outputs/corpus/corpus_0.01-1000.txt

export BC_MODEL_PATH_FOR_RMTK=${BC_PROJECT_BASE_DIR}/models/rlrd

export BC_OUTPUT_BASE_DIR_FOR_CORPUS=${BC_PROJECT_BASE_DIR}/outputs/corpus


SCRATCH=~
export TRANSFORMERS_CACHE=$SCRATCH/.cache/huggingface/transformers \
export HF_DATASETS_CACHE=$SCRATCH/RankCSE/data/ \
export HF_HOME=$SCRATCH/.cache/huggingface \
export XDG_CACHE_HOME=$SCRATCH/.cache 

