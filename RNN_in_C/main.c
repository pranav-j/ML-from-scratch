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

    int EPOCHS = 25;
    int H = 32;
    double LR = 0.01;
    double max_norm = 5.0;

    RNN* rnn = rnn_create(H, c->vocab_size);

    int num_chunks = c->text_length/T;

    for(int epoch = 0; epoch < EPOCHS; epoch++) {
        RNNGradients* grads = gradients_create(H, rnn->V);
        RNNCache* cache = cache_create();
        cache->h_cache[0] = matrix_create(H, 1);
        matrix_init(cache->h_cache[0], 0.0);

        for(int chunk_num = 0; chunk_num < num_chunks; chunk_num++) {
            gradients_zero(grads);

            
            


            int chunk[T + 1];
            // Forward

            for(int i = 0; i < T + 1; i++) {
                unsigned char ch = (unsigned char) c->text[chunk_num*T + i];
                chunk[i] = c->char_to_index[ch];
            }
            double loss = rnn_forward(rnn, cache, chunk);

            if(chunk_num == num_chunks - 1) {
                printf("EPOCHS: %d, Chunk %d loss: %f \n", epoch, chunk_num, loss);
            }
            

            // Backward

            rnn_backward(rnn, cache, chunk, grads);

            // Updation
            gradients_clip(grads, max_norm)
            rnn_update(rnn, grads, LR);
            

            Matrix* cache_carry = matrix_copy(cache->h_cache[T]);
            cache_reset(cache);
            cache->h_cache[0] = cache_carry;
        }

        gradients_free(grads);
        cache_free(cache);
    }
    corpus_free(c);
    return 0;
}
