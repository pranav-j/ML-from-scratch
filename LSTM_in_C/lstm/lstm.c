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

void lstm_step(LSTM* lstm, int x_idx, Matrix* h, Matrix* c) {
    int H = lstm->H;
    Matrix* z = matrix_create(lstm->H + lstm->V, 1);
    matrix_init(z, 0.0);
    for(int i = 0; i < h->rows; i++) {
        z->values[i][0] = h->values[i][0];
    }
    z->values[H + x_idx][0] = 1; // one-hot

    Matrix* W_dot_z = dot(lstm->W, z);
    Matrix* a = add(W_dot_z, lstm->b);

    matrix_free(W_dot_z);

    

    for(int j = 0; j < H; j++) {
        double f = sigmoid(a->values[j][0]);
        double i = sigmoid(a->values[H + j][0]);
        double g = tanh(a->values[2*H + j][0]);
        double o = sigmoid(a->values[3*H + j][0]);

        c->values[j][0] = f * c->values[j][0] + i * g;
        h->values[j][0] = tanh(c->values[j][0]) * o;
    }
    matrix_free(a);
    matrix_free(z);
}

double lstm_output_loss(LSTM* lstm, Matrix* h, int target) {
    Matrix* Wy_dot_h = dot(lstm->Wy, h);
    Matrix* r = add(Wy_dot_h, lstm->by);
    Matrix* prob = softmax(r);
    matrix_free(Wy_dot_h);
    matrix_free(r);
    double loss = -log(prob->values[target][0]);
    matrix_free(prob);
    return loss;
}

double lstm_forward(LSTM* lstm, const int* chunk, Matrix* h, Matrix* c) {
    double loss = 0.0;
    for(int t = 0; t < T; t++) {
        lstm_step(lstm, chunk[t], h, c);
        loss += lstm_output_loss(lstm, h, chunk[t + 1]);
    }
    return loss;
}   