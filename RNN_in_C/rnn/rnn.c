#include "rnn.h"
#include <matrix/matrix.h>
#include <matrix/operations.h>
#include "../data_ops/data.h"
#include <math.h>


Matrix* softmax(Matrix* m) {
    double max_val = m->values[0][0];
    for (int i = 0; i < m->rows; i++)
        if (m->values[i][0] > max_val) max_val = m->values[i][0];

    double total = 0.0;
    Matrix* out = matrix_create(m->rows, m->cols);
    for (int i = 0; i < m->rows; i++) {
        out->values[i][0] = exp(m->values[i][0] - max_val);
        total += out->values[i][0];
    }
    for (int i = 0; i < m->rows; i++) out->values[i][0] /= total;
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

double rnn_forward(RNN* rnn, RNNCache* cache, int* chunk) { // int chunk[] -- same thing
    double loss = 0.0;
    for(int t = 0; t < T; t++) {
        rnn_step(rnn, cache, chunk[t], t);
        int target = chunk[t + 1];
        double lil_loss = -log(cache->p_cache[t]->values[target][0]);
        loss += lil_loss;
    }
    return loss;
}