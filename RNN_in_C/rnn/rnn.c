#include "rnn.h"
#include <matrix/matrix.h>
#include <matrix/operations.h>
#include "../data_ops/data.h"
#include <math.h>
#include <stdlib.h>

static double square(double x) { return x * x; }

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

RNN* rnn_create(int H, int V) {
    RNN* rnn = malloc(sizeof(RNN));

    rnn->H = H;
    rnn->V = V;

    rnn->Wxh = matrix_create(H, V);
    rnn->Whh = matrix_create(H, H);
    rnn->bh = matrix_create(H, 1);
    rnn->Why = matrix_create(V, H);
    rnn->by = matrix_create(V, 1);

    matrix_randomize(rnn->Wxh, H);
    matrix_randomize(rnn->Whh, H);
    matrix_init(rnn->bh, 0.0);
    matrix_randomize(rnn->Why, H);
    matrix_init(rnn->by, 0.0);

    return rnn;
}

void rnn_free(RNN* rnn) {
    if (!rnn) return;

    matrix_free(rnn->Wxh);
    matrix_free(rnn->Whh);
    matrix_free(rnn->bh);
    matrix_free(rnn->Why);
    matrix_free(rnn->by);

    free(rnn);
}

RNNGradients* gradients_create(int H, int V) {
    RNNGradients* grads = malloc(sizeof(RNNGradients));

    grads->dWxh = matrix_create(H, V);
    grads->dWhh = matrix_create(H, H);
    grads->dbh = matrix_create(H, 1);
    grads->dWhy = matrix_create(V, H);
    grads->dby = matrix_create(V, 1);
    gradients_zero(grads);

    return grads;
}

void gradients_zero(RNNGradients* grads) {
    matrix_init(grads->dWxh, 0.0);
    matrix_init(grads->dWhh, 0.0);
    matrix_init(grads->dbh, 0.0);
    matrix_init(grads->dWhy, 0.0);
    matrix_init(grads->dby, 0.0);
}

void gradients_free(RNNGradients* grads) {
    if(!grads) return;

    matrix_free(grads->dWxh);
    matrix_free(grads->dWhh);
    matrix_free(grads->dbh);
    matrix_free(grads->dWhy);
    matrix_free(grads->dby);

    free(grads);
}

RNNCache* cache_create(void) {
    RNNCache* cache = malloc(sizeof(RNNCache));

    for(int t = 0; t <= T; t++) cache->h_cache[t] = NULL;
    for(int t = 0; t < T; t++) cache->x_cache[t] = NULL;
    for(int t = 0; t < T; t++) cache->p_cache[t] = NULL;
    
    return cache;
}

void cache_reset(RNNCache* cache) {
    for(int t = 0; t <= T; t++) {
        if(cache->h_cache[t]) {matrix_free(cache->h_cache[t]); cache->h_cache[t] = NULL;}
    }

    for(int t = 0; t < T; t++) {
        if(cache->x_cache[t]) {matrix_free(cache->x_cache[t]); cache->x_cache[t] = NULL;}
        if(cache->p_cache[t]) {matrix_free(cache->p_cache[t]); cache->p_cache[t] = NULL;}
    }
}

void cache_free(RNNCache* cache) {
    if(!cache) return;
    cache_reset(cache);
    free(cache);
}

void rnn_step(RNN* rnn, RNNCache* cache, int input_index, int t) {
    Matrix* input = one_hot(input_index, rnn->V);
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

void rnn_backward(RNN* rnn, RNNCache* cache, int* chunk, RNNGradients* grads) {
    Matrix* dh_next = matrix_create(rnn->H, 1);
    matrix_init(dh_next, 0.0);

    for(int t = T - 1; t >= 0; t--) {
        Matrix* y_t = one_hot(chunk[t + 1], rnn->V);
        Matrix* dzt = subtract(cache->p_cache[t], y_t);
        matrix_free(y_t);

        Matrix* WhyT = transpose(rnn->Why);
        Matrix* WhyT_dot_dzt = dot(WhyT, dzt);
        Matrix* dh_t = add(WhyT_dot_dzt, dh_next);
        matrix_free(WhyT);
        matrix_free(WhyT_dot_dzt);

        Matrix* ones = matrix_create(rnn->H, 1);
        matrix_init(ones, 1.0);
        Matrix* h_cache_square = apply(square, cache->h_cache[t + 1]);
        Matrix* sub_ones_h_cache_square = subtract(ones, h_cache_square);
        Matrix* da_t = hadamard(dh_t, sub_ones_h_cache_square);
        matrix_free(ones);
        matrix_free(h_cache_square);
        matrix_free(sub_ones_h_cache_square);


        //Gradinet updates
        Matrix* h_cache_next_T = transpose(cache->h_cache[t+1]);
        Matrix* dzt_dot_h_cache_next_T = dot(dzt, h_cache_next_T);
        matrix_add_inplace(grads->dWhy, dzt_dot_h_cache_next_T);
        matrix_free(h_cache_next_T);
        matrix_free(dzt_dot_h_cache_next_T);

        matrix_add_inplace(grads->dby, dzt);

        Matrix* x_cacheT = transpose(cache->x_cache[t]);
        Matrix* da_t_dot_x_cacheT = dot(da_t, x_cacheT);
        matrix_add_inplace(grads->dWxh, da_t_dot_x_cacheT);
        matrix_free(x_cacheT);
        matrix_free(da_t_dot_x_cacheT);

        Matrix* h_cache_prev_T = transpose(cache->h_cache[t]);
        Matrix* da_t_dot_h_cache_prev_T = dot(da_t, h_cache_prev_T);
        matrix_add_inplace(grads->dWhh, da_t_dot_h_cache_prev_T);
        matrix_free(h_cache_prev_T);
        matrix_free(da_t_dot_h_cache_prev_T);

        matrix_add_inplace(grads->dbh, da_t);

        Matrix* WhhT = transpose(rnn->Whh);
        Matrix* new_dh_next = dot(WhhT, da_t);
        matrix_free(WhhT);
        matrix_free(dh_next);
        dh_next = new_dh_next;

        matrix_free(dzt);
        matrix_free(dh_t);
        matrix_free(da_t);
    }
    matrix_free(dh_next);
}