#include "operations.h"
#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>

int check_dimentional_equality(Matrix* m1, Matrix* m2) {
    if(m1->rows == m2->rows && m1->cols == m2->cols) return 1;
    return 0;
}

Matrix* multiply(Matrix* m1, Matrix* m2) {
    if(m1->cols != m2->rows) {
        fprintf(stderr, "matrix_multiply: shape mismatch (%dx%d * %dx%d)\n",
                m1->rows, m1->cols, m2->rows, m2->cols);
        return NULL;
    }

    Matrix* out_mat = matrix_create(m2->rows, m1->cols);
    if(!out_mat) return NULL;

    for( int i = 0; i < m1->rows; i++) {
        for( int k = 0; k < m2->cols; k++) {
            double sum = 0.0;
            for( int j = 0; j < m2->cols; j++) {
                sum += m1->values[i][j] * m2->values[j][i];
            }
            out_mat->values[i][k] = sum;
        }

    }

    return out_mat;
}