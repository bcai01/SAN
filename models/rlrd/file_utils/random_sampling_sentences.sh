
SEED=42
python random_sampling_sentences.py \
    --sentence_file $DATA_DIR/corpus.txt \
    --output_file $OUTPUT_DIR/corpus/corpus_0.01.txt \
    --seed $SEED \
    --n_sentences 10000
