#ifndef LSTM_H
#define LSTM_H

#include <matrix/matrix.h>

#define T 25

typedef struct {
    int H;
    int V;

    Matrix* W;      // 4H × (H+V)   all four gates stacked: f, i, g, o
    Matrix* b;      // 4H × 1

    Matrix* Wy;     // V × H 
    Matrix* by;     // V × 1
} LSTM;

typedef struct {
    Matrix* probs[T];
    Matrix* h[T + 1];
    Matrix* c[T + 1];
    Matrix* gates[T];
    Matrix* z[T];
} LSTMCache;

typedef struct {
    Matrix* dW;      // 4H × (H+V)   all four gates stacked: f, i, g, o
    Matrix* db;      // 4H × 1

    Matrix* dWy;     // V × H 
    Matrix* dby;     // V × 1
} LSTMGradients;

LSTM* lstm_create(int H, int V);
void lstm_free(LSTM* lstm);
LSTMCache* lstm_cache_create(int H, int V);
void lstm_cache_free(LSTMCache* cache);
LSTMGradients* lstm_gradients_create(int H, int V);
void lstm_gradients_zero(LSTMGradients* grads);
void lstm_gradients_free(LSTMGradients* grads);

void lstm_step(LSTM* lstm, LSTMCache* cache, int t, int x_idx);
double lstm_output_loss(LSTM* lstm, LSTMCache* cache, int t, int target);
double lstm_forward(LSTM* lstm, LSTMCache* cache, const int* chunk);

#endif