#include "matrix.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define LINE_LEN 100

Matrix* matrix_create(int rows, int cols) {
    Matrix* matrix = malloc(sizeof(Matrix));
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
    if(!file) {
        perror(file_name);
        return;
    }
    fprintf(file, "%d\n", matrix->rows);
    fprintf(file, "%d\n", matrix->cols);
    for(int i = 0; i < matrix->rows; i++) {
        for(int j = 0; j < matrix->cols; j++) {
            fprintf(file, "%.6f\n", matrix->values[i][j]);
        }
    }
    if(fclose(file) != 0) {
        perror("fopen");
        return;
    }
    printf("Matrix Succesfully saved to %s \n", file_name);
}

Matrix* matrix_load(char* file_name) {
    FILE* file = fopen(file_name, "r");
    if(!file) {
        perror(file_name);
        return NULL;
    }
    char line[LINE_LEN];
    if(!fgets(line, LINE_LEN, file)) { fclose(file); return NULL; };
    int rows = atoi(line);
    if(!fgets(line, LINE_LEN, file)) { fclose(file); return NULL; };
    int cols = atoi(line);

    Matrix* matrix = matrix_create(rows, cols);
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            if(!fgets(line, LINE_LEN, file)) {
                matrix_free(matrix);
                fclose(file);
                return NULL;
            }
            matrix->values[i][j] = strtod(line, NULL);
        }
    }
    if(fclose(file) != 0) {
        perror("Error loading matrix ");
        matrix_free(matrix);
        return NULL;
    }
    printf("Matrix loaded successfully! \n");
    return matrix;
}

// There are issues with this, but fine for now.
double uniform_distribution(double low, double high) {
	double difference = high - low;
	int scale = 10000;
	int scaled_difference = (int)(difference * scale);
	return low + (1.0 * (rand() % scaled_difference) / scale);
}

void matrix_randomize(Matrix* matrix, int n) {
	double min = -1.0 / sqrt(n);
	double max = 1.0 / sqrt(n);
	for (int i = 0; i < matrix->rows; i++) {
		for (int j = 0; j < matrix->cols; j++) {
			matrix->values[i][j] = uniform_distribution(min, max);
		}
	}
}

int matrix_argmax(Matrix* matrix) {
    // Expects a single column matrix.
    double max_value = matrix->values[0][0];
    int max_idx = 0;
    for( int i = 0; i < matrix->rows; i++) {
        if (matrix->values[i][0] > max_value) { 
            max_idx = i; 
            max_value = matrix->values[i][0];
        }
    }
    return max_idx;
}

Matrix* matrix_flatten(Matrix* matrix, int axis) {
    Matrix* out_mat;
    if(axis == 0) {
        out_mat = matrix_create(matrix->rows * matrix->cols, 1);
    } else if(axis == 1) {
        out_mat = matrix_create(1, matrix->rows * matrix->cols);
    } else {
        printf("Axis should be 1 or 0 \n");
        return NULL;
    }
    if(!out_mat) return NULL;

    if(axis == 0) {
        for( int i = 0; i < matrix->rows; i++) {
            for( int j = 0; j < matrix->cols; j++) {
                out_mat->values[i * matrix->cols + j][0] = matrix->values[i][j];
            }
        }
    } else {
        for( int i = 0; i < matrix->rows; i++) {
            for( int j = 0; j < matrix->cols; j++) {
                out_mat->values[0][i * matrix->cols + j] = matrix->values[i][j];
            }
        }
    }


    return out_mat;
}