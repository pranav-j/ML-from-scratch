#include <matrix/matrix.h>
#include "../data_ops/data.h"
#include <math.h>

Matrix* softmax(Matrix* matrix) {
    double total = 0;
    for(int i = 0; i < matrix->rows; i++) {
        for(int j = 0; j < matrix->cols; j++) {
            total += exp(matrix->values[i][j]);
        }
    }

    Matrix* out = matrix_create(matrix->rows, matrix->cols);
    if(!out) return NULL;

    for(int i = 0; i < matrix->rows; i++) {
        for(int j = 0; j < matrix->cols; j++) {
            out->values[i][j] = exp(matrix->values[i][j])/total;
        }
    }

    return out;
}

void rnn_step(RNN* rnn, RNNCache* cache, int input_index, int t, int vocab_size) {
    Matrix* input = one_hot(input_index, vocab_size);
    cache->x_cache[t] = input;

    Matrix* a = add(add(dot(rnn->Wxh, input), dot(rnn->Whh, cache->h_cache[t])), rnn->bh);
    Matrix* h = apply(tanh, a);
    cache->h_cache[t + 1] = h;

    Matrix* z = add(dot(rnn->Why, h), rnn->by);
    Matrix* y = softmax(z);
    cache->p_cache[t] = y;
}

Matrix* forward(RNN* rnn, RNNCache* cache, )