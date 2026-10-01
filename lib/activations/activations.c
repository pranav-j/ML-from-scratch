#include <activations/activations.h>
#include <matrix/matrix.h>

#include <math.h>
#include <stddef.h>

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_prime_from_a(double a) {
    return a * (1.0 - a);
}

double tanh_prime_from_a(double a) {
    return 1.0 - a * a;
}

Matrix* softmax(Matrix* matrix) {
    Matrix* out = matrix_create(matrix->rows, matrix->cols);
    if(!out) return NULL;

    double max_val = matrix->values[0][0];
    for(int i = 0; i < matrix->rows; i++) {
        for(int j = 0; j < matrix->cols; j++) {
            if(matrix->values[i][j] > max_val) max_val = matrix->values[i][j];
        }
    }

    double total = 0.0;
    for(int i = 0; i < matrix->rows; i++) {
        for(int j = 0; j < matrix->cols; j++) {
            out->values[i][j] = exp(matrix->values[i][j] - max_val);
            total += out->values[i][j];
        }
    }

    for(int i = 0; i < matrix->rows; i++) {
        for(int j = 0; j < matrix->cols; j++) {
            out->values[i][j] /= total;
        }
    }

    return out;
}
