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

void lstm_backward(LSTM* lstm, LSTMCache* cache, LSTMGradients* grads, int* chunk) {
    int H = lstm->H;
    Matrix* dh_next = matrix_create(H, 1);
    matrix_init(dh_next, 0.0);
    Matrix* dc_next = matrix_create(H, 1);
    matrix_init(dc_next, 0.0);
    Matrix* da   = matrix_create(4 * H, 1);
    Matrix* Wy_T = transpose(lstm->Wy);
    Matrix* W_T  = transpose(lstm->W);

    for(int t = T - 1; t >= 0; t--) {
        // Stage A: undo the output layer
        Matrix* dr = matrix_copy(cache->probs[t]);
        dr->values[chunk[t+1]][0] -= 1.0;
        Matrix* h_transpose = transpose(cache->h[t+1]);
        Matrix* dr_dot_h_transpose = dot(dr, h_transpose);
        matrix_free(h_transpose);
        matrix_add_inplace(grads->dWy, dr_dot_h_transpose);
        matrix_add_inplace(grads->dby, dr);

        matrix_free(dr_dot_h_transpose);

        Matrix* dh = dot(Wy_T, dr);
        matrix_free(dr);
        matrix_add_inplace(dh, dh_next);

        // Stage B: undo the cell
        for(int j = 0; j < H; j++) {
            double f      = cache->gates[t]->values[j][0];
            double i      = cache->gates[t]->values[H + j][0];
            double g      = cache->gates[t]->values[2 * H + j][0];
            double o      = cache->gates[t]->values[3 * H + j][0];
            double c_prev = cache->c[t]->values[j][0];
            double c      = cache->c[t + 1]->values[j][0];
            double tc     = tanh(c);
            double dh_j   = dh->values[j][0];
            double dcn_j  = dc_next->values[j][0];

            double d_o = dh_j * tc;
            double dc  = dh_j * o * (1.0 - tc * tc) + dcn_j;

            double df = dc * c_prev;
            double di = dc * g;
            double dg = dc * i;

            da->values[j][0]         = df  * f * (1.0 - f);
            da->values[H + j][0]     = di  * i * (1.0 - i);
            da->values[2 * H + j][0] = dg  * (1.0 - g * g);
            da->values[3 * H + j][0] = d_o * o * (1.0 - o);

            dc_next->values[j][0] = dc * f;
        }
        matrix_free(dh);

        // Stage C: undo a = W·z + b
        Matrix* z_T    = transpose(cache->z[t]);
        Matrix* da_z_T = dot(da, z_T);              // 4H×1 · 1×(H+V) = 4H×(H+V), same as W
        matrix_add_inplace(grads->dW, da_z_T);
        matrix_free(z_T);
        matrix_free(da_z_T);

        matrix_add_inplace(grads->db, da);

        Matrix* dz = dot(W_T, da);                  // (H+V)×4H · 4H×1 = (H+V)×1
        for (int j = 0; j < H; j++)
            dh_next->values[j][0] = dz->values[j][0];
        matrix_free(dz);
    }

    matrix_free(Wy_T);
    matrix_free(W_T);  
    matrix_free(da);
    matrix_free(dh_next);
    matrix_free(dc_next);
}

void lstm_update(LSTM* lstm, LSTMGradients* grads, double lr) {
    matrix_update(lstm->W, grads->dW, lr);
    matrix_update(lstm->b, grads->db, lr);
    matrix_update(lstm->Wy, grads->dWy, lr);
    matrix_update(lstm->by, grads->dby, lr);
}