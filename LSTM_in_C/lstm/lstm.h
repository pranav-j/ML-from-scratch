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

LSTM* lstm_create(int H, int V);
void lstm_free(LSTM* lstm);
LSTMCache* lstm_cache_create(int H, int V);
void lstm_cache_free(LSTMCache* cache);

void lstm_step(LSTM* lstm, int x_idx, Matrix* h, Matrix* c);
double lstm_output_loss(LSTM* lstm, Matrix* h, int target);
double lstm_forward(LSTM* lstm, const int* chunk, Matrix* h, Matrix* c);

#endif