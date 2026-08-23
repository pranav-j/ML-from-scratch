typedef struct {
    double** values;
    int rows;
    int cols;
} Matrix;
Matrix* matrix_create(int rows, int cols);
void matrix_init(Matrix* matrix, int n);
void matrix_free(Matrix* matrix);
void matrix_print(Matrix* matrix);
Matrix* matrix_copy(Matrix* matrix);
void matrix_save(Matrix* matrix, char* file_name);
Matrix* matrix_load(char* file_name);