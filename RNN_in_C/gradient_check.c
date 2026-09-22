// gradient_check.c
#include <matrix/matrix.h>
#include <matrix/operations.h>
#include "rnn/rnn.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Run forward on the current parameters and return the total loss.
// Caller owns the cache; we reset it, set h[0]=0, then forward.
static double run_forward(RNN* rnn, RNNCache* cache, int* chunk) {
    cache_reset(cache);
    cache->h_cache[0] = matrix_create(rnn->H, 1);
    matrix_init(cache->h_cache[0], 0.0);
    return rnn_forward(rnn, cache, chunk);
}

// Numerical gradient of one entry of one parameter matrix.
static double numerical_grad(RNN* rnn, RNNCache* cache, int* chunk,
                              Matrix* param, int i, int j) {
    const double eps = 1e-5;
    double saved = param->values[i][j];

    param->values[i][j] = saved + eps;
    double L_plus = run_forward(rnn, cache, chunk);

    param->values[i][j] = saved - eps;
    double L_minus = run_forward(rnn, cache, chunk);

    param->values[i][j] = saved;   // restore
    return (L_plus - L_minus) / (2.0 * eps);
}

static double relative_error(double a, double n) {
    double denom = fmax(1.0, fmax(fabs(a), fabs(n)));
    return fabs(a - n) / denom;
}

// Check a few entries of one parameter matrix. Print each comparison.
static void check_param(const char* name, Matrix* param, Matrix* grad,
                         RNN* rnn, RNNCache* cache, int* chunk) {
    printf("\n--- %s (%d x %d) ---\n", name, param->rows, param->cols);
    // Pick up to 5 entries spread across the matrix
    int checks = 0;
    int total = param->rows * param->cols;
    int stride = total / 5; if (stride < 1) stride = 1;

    for (int k = 0; k < total && checks < 5; k += stride, checks++) {
        int i = k / param->cols;
        int j = k % param->cols;
        double analytical = grad->values[i][j];
        double numerical = numerical_grad(rnn, cache, chunk, param, i, j);
        double err = relative_error(analytical, numerical);
        printf("  [%d][%d]  analytical=% .6e  numerical=% .6e  rel_err=%.2e  %s\n",
               i, j, analytical, numerical, err,
               err < 1e-5 ? "OK" : (err < 1e-3 ? "SUSPECT" : "WRONG"));
    }
}

int main(void) {
    srand(42);

    int H = 3;
    int V = 2;
    // Chunk length is T+1 = 5. T comes from rnn.h.
    int chunk[T + 1] = {0, 1, 0, 1, 0};   // arbitrary tiny sequence

    RNN* rnn = rnn_create(H, V);
    RNNCache* cache = cache_create();
    RNNGradients* grads = gradients_create(H, V);

    // Analytical: one forward, one backward
    run_forward(rnn, cache, chunk);
    gradients_zero(grads);
    rnn_backward(rnn, cache, chunk, grads);

    // Numerical: check each parameter matrix
    check_param("Wxh", rnn->Wxh, grads->dWxh, rnn, cache, chunk);
    check_param("Whh", rnn->Whh, grads->dWhh, rnn, cache, chunk);
    check_param("bh",  rnn->bh,  grads->dbh,  rnn, cache, chunk);
    check_param("Why", rnn->Why, grads->dWhy, rnn, cache, chunk);
    check_param("by",  rnn->by,  grads->dby,  rnn, cache, chunk);

    cache_free(cache);
    gradients_free(grads);
    rnn_free(rnn);
    return 0;
}