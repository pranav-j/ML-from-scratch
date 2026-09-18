#ifndef RNN_H
#define RNN_H

#include <matrix/matrix.h>

#define T 25

typedef struct {
    int V; // vocab_size
    int H;
    Matrix* Wxh;
    Matrix* Whh;
    Matrix* bh;
    Matrix* Why;
    Matrix* by;
} RNN;

typedef struct {
    Matrix* dWxh;
    Matrix* dWhh;
    Matrix* dbh;
    Matrix* dWhy;
    Matrix* dby;
} RNNGradients;

typedef struct {
    Matrix* h_cache[T + 1];
    Matrix* x_cache[T];
    Matrix* p_cache[T];
} RNNCache;

RNN* rnn_create(int H, int V);
void rnn_free(RNN* rnn);

RNNGradients* gradients_create(int H, int V);
void gradients_free(RNNGradients* grads);
void gradients_zero(RNNGradients* grads);

RNNCache* cache_create(void);
void cache_free(RNNCache* cache);
void cache_reset(RNNCache* cache);

Matrix* softmax(Matrix* matrix);
void rnn_step(RNN* rnn, RNNCache* cache, int input_index, int t);
double rnn_forward(RNN* rnn, RNNCache* cache, int* chunk);

#endif