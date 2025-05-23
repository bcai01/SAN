OUTPUT_DIR=$PROJECT_DIR/outputs

SEED=61507
CUDA_VISIBLE_DEVICES=0 python get_sncse_embedding.py \
    --checkpoint $BC_PERTRAINED_MODEL_BASE_DIR/sncse-roberta-base-rank \
    --corpus_file $BC_OUTPUT_RANKENCODER_CORPUS_10000_PATH \
    --sentence_vectors_np_file ${BC_PROJECT_BASE_DIR}/corpus/sncse-roberta-base-corpus_0.01_sncse.npy
