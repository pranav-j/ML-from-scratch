#include "lstm.h"
#include <matrix/matrix.h>
#include <stdlib.h>

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