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