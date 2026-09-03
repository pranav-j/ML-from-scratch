#include "../matrix/matrix.h"

typedef struct {
    int num_inputs;
    int num_hiddens;
    int num_outputs;
    double learning_rate;
    Matrix* W1;     // (num_hiddens, num_inputs)
    Matrix* b1;     // (num_hidden, 1)
    Matrix* W2;     // (num_outputs, num_hiddens)
    Matrix* b2;     // (num_outputs, 1)
} NeuraNetwork;

NeuralNetwork* nn_create(int num_inputs, int num_hiddens, int num_outputs, double learning_rate);
void nn_free(NeuralNetwork* nn);
void forward(NeuralNetwork* nn, Matrix* x, Matrix** z1, Matrix** a1, Matrix** z2, Matrix** a2);
double nn_train_one(NeuralNetwork* nn, Matrix* x, Matrix* y);