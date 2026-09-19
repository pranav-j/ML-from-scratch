#include <matrix/matrix.h>
#include <matrix/operations.h>
#include "data_ops/data.h"
#include "rnn/rnn.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


int main(void) {
    Corpus* c = corpus_load("data/corpus.txt");
    printf("Text length %d \n", c->text_length);
    printf("Vocab size %d \n", c->vocab_size);

    int H = 32;

    RNN* rnn = rnn_create(H, c->vocab_size);

    int num_chunks = c->text_length/T;

    for(int chunk_num = 0; chunk_num < num_chunks; chunk_num++) {
        int chunk[T + 1];
        RNNCache* cache = cache_create();
        cache->h_cache[0] = matrix_create(H, 1);
        matrix_init(cache->h_cache[0], 0.0);


        for(int i = 0; i < T + 1; i++) {
            unsigned char ch = (unsigned char) c->text[chunk_num*T + i];
            chunk[i] = c->char_to_index[ch];
        }
        double loss = rnn_forward(rnn, cache, chunk);
        printf("Chunk %d loss: %f \n", chunk_num, loss);
        cache_free(cache);
    }


    corpus_free(c);
    return 0;
}
