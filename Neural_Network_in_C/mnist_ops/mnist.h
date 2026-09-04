#ifndef MNIST_H
#define MNIST_H

#include "../matrix/matrix.h"

typedef struct {
    Matrix* input;    // (784, 1) — normalized to [0, 1]
    Matrix* target;   // (10, 1)  — one-hot
    int     label;    // 0..9 (for convenience during evaluation)
} MnistSample;

typedef struct {
    MnistSample* samples;
    int count;
} MnistData;

// Load up to `max_samples` samples from a Kaggle-style MNIST CSV
// (first row is header; each data row is `label,px0,px1,...,px783`).
// Pass max_samples <= 0 to load the whole file.
MnistData* mnist_load(const char* path, int max_samples);
void       mnist_free(MnistData* data);

#endif