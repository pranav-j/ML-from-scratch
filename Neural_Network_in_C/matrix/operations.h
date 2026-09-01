int check_dimentional_equality(Matrix* m1, Matrix* m2);
Matrix* dot(Matrix* m1, Matrix* m2);
Matrix* hadamard(Matrix* m1, Matrix* m2)
Matrix* add(Matrix* m1, Matrix* m2);
Matrix* subtract(Matrix* m1, Matrix* m2);
Matrix* scale(Matrix* matrix, double n);
Matrix* add_scalar(Matrix* matrix, double n);
Matrix* transpose(Matrix* matrix);
Matrix* apply(double (*func)(double), Matrix* matrix);
Matrix* matrix_update(Matrix* matrix, Matrix* gradient, double learning_rate);