#include <stdio.h>
#include <stdlib.h>

int main() {
    FILE * file = fopen("../data.csv", "r");
    if(file != NULL) {
        printf("File found! \n");
    }

    char line[256];
    int capacity = 10;
    int size = 0;

    double * x = malloc(capacity * sizeof(double));
    double * y = malloc(capacity * sizeof(double));

    if(x == NULL || y == NULL) {
        printf("Memory allocation failed! \n");
        free(x);
        free(y);
        return 1;
    }

    while(fgets(line, sizeof(line), file) != NULL) {
        if(size == capacity) {
            capacity *= 2;
            double * x_temp = realloc(x, capacity * sizeof(double));
            double * y_temp = realloc(y, capacity * sizeof(double));

            if(x_temp == NULL || y_temp == NULL) {
                printf("Memory reallocation failed! \n");

                free(x_temp);
                free(y_temp);

                return 1;
            }

            x = x_temp;
            y = y_temp;
        }

        if(sscanf(line, "%lf,%lf", &x[size], &y[size])) {
            size++;
        }
    }

    // for(int i = 0; i < size; i++){
    //     printf("In ROW %d X : %f, Y : %f \n", i, x[i], y[i]);
    // }

    // y = wx + b
    // MSE = ((wx + b - y)**2)/N
    // Gradient of MSE wrt w = (2 * x * (wx + b - y))/N
    // Gradient of MSE wrt b = (2 * (wx + b - y))/N

    double w = 0, b = 0;
    int epochs = 100;
    int batch_size = 5;
    int batches = size/batch_size;
    double learning_rate = 0.0001;

    printf("Batches : %d", batches);

    for(int i = 0; i < epochs; i++) {
        for(int j = 0; j < batches; j++) {
            double w_grad_sum = 0;
            double b_grad_sum = 0;

            for(int k = j * batch_size; k < (j + 1) * batch_size; k++) {
                double w_grad = (2 * x[k] * (w * x[k] + b - y[k]));
                double b_grad = (2 * (w * x[k] + b - y[k]));

                w_grad_sum += w_grad;
                b_grad_sum += b_grad;
            }

            double w_grad_mean = w_grad_sum/batch_size;
            double b_grad_mean = b_grad_sum/batch_size;

            w -= learning_rate * w_grad_mean;
            b -= learning_rate * b_grad_mean;
        }

        printf("Epoch %d, W = %f, B = %f \n", i, w, b);
    }

    return 0;
}