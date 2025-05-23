# DATA_DIR=$PROJECT_DIR/data
# OUTPUT_DIR=$PROJECT_DIR/outputs

SEED=42
python random_sampling_sentences.py \
    --sentence_file $BC_WIKI1M_PATH \
    --output_file $BC_OUTPUT_BASE_DIR_FOR_CORPUS/corpus_0.01-1000.txt \
    --seed $SEED \
    --n_sentences 10000
