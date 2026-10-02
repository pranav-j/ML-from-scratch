#include "lstm.h"
#include <matrix/matrix.h>
#include <matrix/operations.h>
#include <activations/activations.h>
#include <stdlib.h>
#include <math.h>

LSTM* lstm_create(int H, int V) {
    LSTM* lstm = malloc(sizeof(LSTM));

    if(!lstm) return NULL;

    lstm->H = H;
    lstm->V = V;

    lstm->W = matrix_create(4*H, H+V);
    lstm->b = matrix_create(4*H, 1);
    lstm->Wy = matrix_create(V, H);
    lstm->by = matrix_create(V, 1);

    matrix_randomize(lstm->W, H+V);
    matrix_init(lstm->b, 0.0);
    matrix_randomize(lstm->Wy, H);
    matrix_init(lstm->by, 0.0);

    return lstm;
}

void lstm_free(LSTM* lstm) {
    if(!lstm) return;

    matrix_free(lstm->W);
    matrix_free(lstm->b);
    matrix_free(lstm->Wy);
    matrix_free(lstm->by);

    free(lstm);
}

LSTMCache* lstm_cache_create(int H, int V) {
    LSTMCache* cache = malloc(sizeof(LSTMCache));
    if(!cache) return NULL;

    for(int t = 0; t <= T; t++) {cache->h[t] = matrix_create(H, 1); matrix_init(cache->h[t], 0.0);}
    for(int t = 0; t <= T; t++) {cache->c[t] = matrix_create(H, 1); matrix_init(cache->c[t], 0.0);}
    for(int t = 0; t < T; t++) {cache->probs[t] = matrix_create(V, 1); matrix_init(cache->probs[t], 0.0);}
    for(int t = 0; t < T; t++) {cache->gates[t] = matrix_create(4*H, 1); matrix_init(cache->gates[t], 0.0);}
    for(int t = 0; t < T; t++) {cache->z[t] = matrix_create(H + V, 1); matrix_init(cache->z[t], 0.0);}

    return cache;
}

void lstm_cache_free(LSTMCache* cache) {
    if(!cache) return;

    for(int t = 0; t <= T; t++) matrix_free(cache->h[t]);
    for(int t = 0; t <= T; t++) matrix_free(cache->c[t]);
    for(int t = 0; t < T; t++) matrix_free(cache->probs[t]);
    for(int t = 0; t < T; t++) matrix_free(cache->gates[t]);
    for(int t = 0; t < T; t++) matrix_free(cache->z[t]);

    free(cache);
}

void lstm_cache_carry(LSTMCache* cache) {
    for(int i = 0; i < cache->h[0]->rows; i++) {
        cache->h[0]->values[i][0] = cache->h[T]->values[i][0];
        cache->c[0]->values[i][0] = cache->c[T]->values[i][0];
    }
}

void lstm_cache_reset_state(LSTMCache* cache) {
    matrix_init(cache->h[0], 0.0);
    matrix_init(cache->c[0], 0.0);
}

LSTMGradients* lstm_gradients_create(int H, int V) {
    LSTMGradients* grads = malloc(sizeof(LSTMGradients));
    if(!grads) return NULL;

    grads->dW = matrix_create(4*H, H+V);
    grads->db = matrix_create(4*H, 1);
    grads->dWy = matrix_create(V, H);
    grads->dby = matrix_create(V, 1);

    lstm_gradients_zero(grads);

    return grads;
}

void lstm_gradients_zero(LSTMGradients* grads) {
    if(!grads) return;

    matrix_init(grads->dW, 0.0);
    matrix_init(grads->db, 0.0);
    matrix_init(grads->dWy, 0.0);
    matrix_init(grads->dby, 0.0);
}

void lstm_gradients_free(LSTMGradients* grads) {
    if(!grads) return;

    matrix_free(grads->dW);
    matrix_free(grads->db);
    matrix_free(grads->dWy);
    matrix_free(grads->dby);

    free(grads);
}

void lstm_step(LSTM* lstm, LSTMCache* cache, int t, int x_idx) {
    int H = lstm->H;
    matrix_init(cache->z[t], 0.0);
    for(int i = 0; i < cache->h[t]->rows; i++) {
        cache->z[t]->values[i][0] = cache->h[t]->values[i][0];
    }
    cache->z[t]->values[H + x_idx][0] = 1; // one-hot

    Matrix* W_dot_z = dot(lstm->W, cache->z[t]);
    Matrix* a = add(W_dot_z, lstm->b);

    matrix_free(W_dot_z);

    

    for(int j = 0; j < H; j++) {
        double f = sigmoid(a->values[j][0]);
        double i = sigmoid(a->values[H + j][0]);
        double g = tanh(a->values[2*H + j][0]);
        double o = sigmoid(a->values[3*H + j][0]);

        cache->gates[t]->values[j][0] = f;
        cache->gates[t]->values[H + j][0] = i;
        cache->gates[t]->values[2*H + j][0] = g;
        cache->gates[t]->values[3*H + j][0] = o;

        cache->c[t+1]->values[j][0] = f * cache->c[t]->values[j][0] + i * g;
        cache->h[t+1]->values[j][0] = tanh(cache->c[t+1]->values[j][0]) * o;
    }
    matrix_free(a);
}

double lstm_output_loss(LSTM* lstm, LSTMCache* cache, int t, int target) {
    Matrix* Wy_dot_h = dot(lstm->Wy, cache->h[t+1]);
    Matrix* r = add(Wy_dot_h, lstm->by);
    Matrix* prob = softmax(r);
    for(int i = 0; i < prob->rows; i++) {
        cache->probs[t]->values[i][0] = prob->values[i][0];
    }
    matrix_free(Wy_dot_h);
    matrix_free(r);
    double loss = -log(prob->values[target][0]);
    matrix_free(prob);
    return loss;
}

double lstm_forward(LSTM* lstm, LSTMCache* cache, const int* chunk) {
    double loss = 0.0;
    for(int t = 0; t < T; t++) {
        lstm_step(lstm, cache, t, chunk[t]);
        loss += lstm_output_loss(lstm, cache, t, chunk[t + 1]);
    }
    return loss;
}