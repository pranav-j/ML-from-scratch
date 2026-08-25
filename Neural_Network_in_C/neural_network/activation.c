#include "activation.h"
#include "../matrix/matrix.h"
#include "../matrix/operations.h"

#include <math.h>

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

Matrix* sigmoidPrime(Matrix* input) {
    // Expects sigmoid as input. 
    // σ′(x)=σ(x)⋅(1−σ(x))
    Matrix* ones = matrix_create(input->rows, input->cols);
    matrix_init(ones, 1);
    Matrix* subtracted = subtract(ones, input);
    Matrix* multiplied = multiply(input, subtracted);
    matrix_free(subtracted);
    matrix_free(multiplied);
    return multiplied;
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