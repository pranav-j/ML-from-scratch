#include "operations.h"
#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>

int check_dimentional_equality(Matrix* m1, Matrix* m2) {
    if(m1->rows == m2->rows && m1->cols == m2->cols) return 1;
    return 0;
}

Matrix* dot(Matrix* m1, Matrix* m2) {
    if(m1->cols != m2->rows) {
        fprintf(stderr, "matrix multiply: shape mismatch (%dx%d * %dx%d)\n",
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

Matrix* multiply(Matrix* m1, Matrix* m2) {
    if(!check_dimentional_equality(m1, m2)) {
        fprintf(stderr, "matrix add: shape mismatch (%dx%d + %dx%d)\n",
                m1->rows, m1->cols, m2->rows, m2->cols);
        return NULL;
    }

    Matrix* out = matrix_create(m1->rows, m1->cols);
    if(!out) return NULL;

    for(int i = 0; i < m1->rows; i++) {
        for(int j = 0; j < m1->cols; j++) {
            out->values[i][j] = m1->values[i][j] * m2->values[i][j];
        }
    }

    return out;
}

Matrix* add(Matrix* m1, Matrix* m2) {
    if(!check_dimentional_equality(m1, m2)) {
        fprintf(stderr, "matrix add: shape mismatch (%dx%d + %dx%d)\n",
                m1->rows, m1->cols, m2->rows, m2->cols);
        return NULL;
    }

    Matrix* out = matrix_create(m1->rows, m1->cols);
    if(!out) return NULL;

    for(int i = 0; i < m1->rows; i++) {
        for(int j = 0; j < m1->cols; j++) {
            out->values[i][j] = m1->values[i][j] + m2->values[i][j];
        }
    }

    return out;
}

Matrix* subtract(Matrix* m1, Matrix* m2) {
    if(!check_dimentional_equality(m1, m2)) {
        fprintf(stderr, "matrix subtract: shape mismatch (%dx%d - %dx%d)\n",
                m1->rows, m1->cols, m2->rows, m2->cols);
        return NULL;
    }

    Matrix* out = matrix_create(m1->rows, m1->cols);
    if(!out) return NULL;

    for(int i = 0; i < m1->rows; i++) {
        for(int j = 0; j < m1->cols; j++) {
            out->values[i][j] = m1->values[i][j] - m2->values[i][j];
        }
    }

    return out;
}

Matrix* scale(Matrix* matrix, double n) {
    Matrix* out = matrix_create(matrix->rows, matrix->cols);
    if(!out) return NULL;

    for(int i = 0; i < matrix->rows; i++) {
        for(int j = 0; j< matrix-> cols; j++) {
            out->values[i][j] = matrix->values[i][j] * n;
        }
    }
    return out;
}

Matrix* add_scalar(Matrix* matrix, double n) {
    Matrix* out = matrix_create(matrix->rows, matrix->cols);
    if(!out) return NULL;
    
    for(int i = 0; i < matrix->rows; i++) {
        for(int j = 0; j< matrix-> cols; j++) {
            out->values[i][j] = matrix->values[i][j] + n;
        }
    }
    return out;
}

Matrix* transpose(Matrix* matrix) {
    Matrix* out = matrix_create(matrix->rows, matrix->cols);
    if(!out) return NULL;
    
    for(int i = 0; i < matrix->rows; i++) {
        for(int j = 0; j< matrix-> cols; j++) {
            out->values[j][i] = matrix->values[i][j];
        }
    }
    return out;
}

Matrix* apply(double (*func)(double), Matrix* matrix) {
    Matrix* out = matrix_create(matrix->rows, matrix->cols);
    if(!out) return NULL;
    
    for(int i =0; i < matrix->rows; i++) {
        for(int j = 0; j < matrix->cols; j++) {
            out->values[i][j] = (*func)(matrix->values[i][j]);
            // out->values[i][j] = func(matrix->values[i][j]);      //Same as above.
        }
    }
    return out;
}