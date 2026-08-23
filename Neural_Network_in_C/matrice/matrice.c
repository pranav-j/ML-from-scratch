Matrix* matrix_create(int rows, int cols) {
    Matrix* matrix;
    matrix->rows = rows;
    matrix->cols = cols;
    matrix->values = malloc(rows * sizeof(double*));
    for(int i = 0; i < rows; i++) {
        matrix->values[i] = malloc(cols * sizeof(double));
    }
    return matrix;
}

void matrix_init(Matrix* matrix, double n) {
    for(int i = 0; i < matrix->rows; i++) {
        for(int j = 0; j < matrix->cols; j++) {
            matrix->values[i][j] = n;
        }
    }
}

void matrix_free(Matrix* matrix) {
    for(int i = 0; i < matrix->rows ; i++) {
        free(matrix->values[i]);
    }
    free(matrix->values);
    free(matrix);
}

void matrix_print(Matrix* matrix) {
    for(int i = 0; i < matrix->rows; i++) {
        for(int j = 0; j < matrix->cols; j++) {
            printf("Value at [%d][%d] is %1.3f", i, j, matrix->values[i][j]);
        }
        printf("\n");
    }
}

Matrix* matrix_copy(Matrix* matrix) {
    Matrix* copy_matrix = matrix_create(matrix->rows, matrix->cols);
    for(int i = 0; i < matrix->rows; i++) {
        for(int j = 0; j < matrix->cols; j++) {
            copy_matrix->values[i][j] = matrix->values[i][j];
        }
    }
    return copy_matrix;
}

void matrix_save(Matrix* matrix, char* file_name) {
    FILE* file = fopen(file_name, "w");
    fprintf(file, "%d\n", matrix->rows);
    fprintf(file, "%d\n", matrix->cols);
    for(int i = 0; i < matrix->rows; i++) {
        for(int j = 0; j < matrix->cols; j++) {
            fprintf(file, "%.6f\n", matrix->values[i][j]);
        }
    }
    printf("NN Succesfully saved to %s \n", file_name);
}