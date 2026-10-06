// gradient_check.c — standalone binary, NOT part of training.
// Run once to prove lstm_backward matches numerical gradients.
//
// For every entry of every parameter:
//   numerical  = (loss(w + EPS) - loss(w - EPS)) / (2 * EPS)
//   analytical = what lstm_backward wrote into the gradient struct
// They should agree to ~1e-7 relative error.

#include <matrix/matrix.h>
#include "lstm/lstm.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define EPS 1e-5

// Every forward starts from the same zero state, so the only difference
// between the +EPS and -EPS runs is the nudge itself.
static double run_forward(LSTM* lstm, LSTMCache* cache, const int* chunk) {
    lstm_cache_reset_state(cache);
    return lstm_forward(lstm, cache, chunk);
}

static double numerical_grad(Matrix* param, int r, int col,
                             LSTM* lstm, LSTMCache* cache, const int* chunk) {
    double original = param->values[r][col];

    param->values[r][col] = original + EPS;
    double loss_plus = run_forward(lstm, cache, chunk);

    param->values[r][col] = original - EPS;
    double loss_minus = run_forward(lstm, cache, chunk);

    param->values[r][col] = original;   // restore, or later checks use a perturbed model
    return (loss_plus - loss_minus) / (2.0 * EPS);
}

// W and b have 4H rows in gate blocks f, i, g, o. Labelling each row
// tells you WHICH gate is broken if a block fails.
static const char* gate_of_row(int r, int H) {
    static const char* names[] = { "f", "i", "g", "o" };
    return names[r / H];
}

// Checks every entry (the model is tiny, so this is cheap). Returns # of WRONG entries.
static int check_param(const char* name, Matrix* param, Matrix* grad, int gated,
                       LSTM* lstm, LSTMCache* cache, const int* chunk) {
    int wrong = 0;
    double worst = 0.0;

    printf("\n%s (%d x %d)\n", name, param->rows, param->cols);
    for (int r = 0; r < param->rows; r++) {
        for (int col = 0; col < param->cols; col++) {
            double a = grad->values[r][col];
            double n = numerical_grad(param, r, col, lstm, cache, chunk);
            double diff = fabs(a - n);
            double denom = fabs(a) + fabs(n);
            double rel = (denom > 0.0) ? diff / denom : 0.0;

            // Both values ~0 makes rel_err meaningless noise, so tiny
            // absolute differences count as OK.
            const char* status;
            if (rel < 1e-5 || diff < 1e-9) status = "OK";
            else if (rel < 1e-3)           status = "SUSPECT";
            else                           { status = "WRONG"; wrong++; }

            // Track worst only where gradients are big enough for rel_err to mean something.
            if (denom > 1e-7 && rel > worst) worst = rel;

            printf("  [%2d][%d]%s%s  analytical=% .6e  numerical=% .6e  rel_err=%.2e  %s\n",
                   r, col,
                   gated ? "  gate=" : "", gated ? gate_of_row(r, lstm->H) : "",
                   a, n, rel, status);
        }
    }
    printf("  -> worst rel_err %.2e, %d WRONG\n", worst, wrong);
    return wrong;
}

int main(void) {
    srand(42);   // same "random" weights every run, so failures are reproducible

    int H = 3;   // tiny model: every check runs in milliseconds
    int V = 2;

    LSTM* lstm = lstm_create(H, V);
    LSTMCache* cache = lstm_cache_create(H, V);
    LSTMGradients* grads = lstm_gradients_create(H, V);
    if (!lstm || !cache || !grads) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }

    // Fill all T+1 entries with a fixed pseudo-random 0/1 sequence.
    int chunk[T + 1];
    for (int k = 0; k < T + 1; k++) chunk[k] = rand() % V;

    // Analytical gradients: one forward, then backward on that exact cache.
    double loss = run_forward(lstm, cache, chunk);
    printf("loss = %f\n", loss);
    lstm_gradients_zero(grads);
    lstm_backward(lstm, cache, grads, chunk);

    // Numerical checks. These run many more forwards (overwriting the cache),
    // which is fine: the analytical gradients are already stored in grads.
    int wrong = 0;
    wrong += check_param("W",  lstm->W,  grads->dW,  1, lstm, cache, chunk);
    wrong += check_param("b",  lstm->b,  grads->db,  1, lstm, cache, chunk);
    wrong += check_param("Wy", lstm->Wy, grads->dWy, 0, lstm, cache, chunk);
    wrong += check_param("by", lstm->by, grads->dby, 0, lstm, cache, chunk);

    printf("\n%s\n", wrong == 0 ? "ALL GRADIENTS OK" : "GRADIENT CHECK FAILED");

    lstm_gradients_free(grads);
    lstm_cache_free(cache);
    lstm_free(lstm);
    return wrong == 0 ? 0 : 1;
}