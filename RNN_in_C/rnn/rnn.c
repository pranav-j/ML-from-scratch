#include <matrix/matrix.h>
#include <matrix/operations.h>
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

void rnn_step(RNN* rnn, RNNCache* cache, int input_index, int t) {
    Matrix* input = one_hot(input_index, rnn->vocab_size);
    cache->x_cache[t] = input;

    Matrix* xh = dot(rnn->Wxh, input);
    Matrix* hh = dot(rnn->Whh, cache->h_cache[t]);
    Matrix* sum1 = add(xh, hh);
    Matrix* a = add(sum1, rnn->bh);

    matrix_free(xh);
    matrix_free(hh);
    matrix_free(sum1);

    Matrix* h = apply(tanh, a);
    matrix_free(a);
    cache->h_cache[t + 1] = h;

    Matrix* hy = dot(rnn->Why, h);
    Matrix* z = add(hy, rnn->by);
    matrix_free(hy);
    Matrix* y = softmax(z);
    matrix_free(z);
    cache->p_cache[t] = y;
}

