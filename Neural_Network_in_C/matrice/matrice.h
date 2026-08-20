typedef struct {
    double** values;
    int rows;
    int cols;
} Matrice;
Matrice* matrice_create(int rows, int cols);
void matrice_init(Matrice* matrice, int n);
void matrice_free(Matrice* matrice);
void matrice_print (Matrice* matrice);