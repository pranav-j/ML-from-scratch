#include <matrix/matrix.h>
#include <data_ops/data.h>
#include "lstm/lstm.h"

#include <stdio.h>
#include <stdlib.h>

int main(void) {
    Corpus* corpus = corpus_load(CORPUS_PATH);
    if(!corpus) return 1;

    printf("Text length %d \n", corpus->text_length);
    printf("Vocab size %d \n", corpus->vocab_size);

    LSTM* lstm = lstm_create(100, corpus->vocab_size);
    int num_chunks = (corpus->text_length - 1)/T;
    LSTMCache* cache = lstm_cache_create(lstm->H, lstm->V);
    int epochs = 1;

    for(int epoch = 0; epoch < epochs; epoch++) {
        lstm_cache_reset_state(cache);
        for(int chunk_num = 0; chunk_num<num_chunks; chunk_num++) {
            int chunk[T + 1];
            for(int i = 0; i < T + 1; i++) {
                unsigned char ch = corpus->text[chunk_num * T + i];
                chunk[i] = corpus->char_to_index[ch];
            }
            double loss = lstm_forward(lstm, cache, chunk);
            lstm_cache_carry(cache);
            printf("Loss at chunk_num %d : %f \n", chunk_num, loss);
        }
    }

    lstm_free(lstm);
    lstm_cache_free(cache);
    corpus_free(corpus);
    return 0;
}
