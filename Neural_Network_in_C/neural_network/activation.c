#include "activation.h"
#include "../matrix/matrix.h"
#include "../matrix/operations.h"

#include <math.h>
#include <stdlib.h>

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_prime_from_a(double a) {
    return a * (1.0 - a);
}

Matrix* softmax(Matrix* matrix) {
    double total = 0;
    for(int i = 0; i < matrix->rows; i++) {
        for(int j = 0; j < matrix->cols; j++) {
            total += exp(matrix->values[i][j]);
        }
    }

    Matrix* out = matrix_create(matrix->rows, matrix->cols);
    if(!out) return NULL;

    for(int i = 0; i < matrix->rows; i++) {
        for(int j = 0; j < matrix->cols; j++) {
            out->values[i][j] = exp(matrix->values[i][j])/total;
        }
    }

    return out;
}