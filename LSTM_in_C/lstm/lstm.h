#ifndef LSTM_H
#define LSTM_H

#include <matrix/matrix.h>

typedef struct {
    int H;
    int V;

    Matrix* W;      // 4H × (H+V)   all four gates stacked: f, i, g, o
    Matrix* b;      // 4H × 1

    Matrix* Wy;     // V × H 
    Matrix* by;     // V × 1
} LSTM;

LSTM* lstm_create(int H, int V);
void lstm_free(LSTM* lstm);
void lstm_step(LSTM* lstm, int x_idx, Matrix* h, Matrix* c);

#endif