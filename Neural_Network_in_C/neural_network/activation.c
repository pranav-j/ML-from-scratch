#include "activation.h"
#include "../matrix/matrix.h"
#include "../matrix/operations.h"

#include <math.h>

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

Matrix* sigmoidPrime(Matrix* input) {
    Matrix* ones = matrix_create(input->rows, input->cols);
    matrix_init(ones, 1);
    Matrix* subtracted = subtract(ones, input);
    Matrix* multiplied = multiply(input, subtracted);
    matrix_free(subtracted);
    matrix_free(multiplied);
    return multiplied;
}

