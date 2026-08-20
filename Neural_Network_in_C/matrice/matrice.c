Matrice* matrice_create(int rows, int cols) {
    Matrice matrice;
    matrice->rows = rows;
    matrice->cols = cols;
    matrice->values = malloc(rows * sizeof(double*));
    for(int i = 0; i < rows; i++) {
        matrice->values[i] = malloc(cols * sizeof(double));
    }
    return matrice;
}

void matrice_init(Matrice* matrice, int n) {
    for(int i = 0; i < matrice->rows; i++) {
        for(int j = 0; j < matrice->cols; j++) {
            matrice->values[i][j] = n;
        }
    }
}

void matrice_free(Matrice* matrice) {
    for(int i = 0; i < matrice->rows ; i++) {
        free(matrice->values[i]);
    }
    free(matrice->values);
    free(matrice);
    matrice = NULL;
}

void matrice_print (Matrice* matrice) {
    for(int i = 0; i < matrice->rows; i++) {
        for(int j = 0; j < matrice->cols; j++) {
            printf("Value at [%d][%d] is %1.3f", i, j, matrice->values[i][j]);
        }
        printf("\n");
    }
}

