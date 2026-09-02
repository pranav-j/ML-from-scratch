#include "nn.h"
#include "activation.h"
#include "../matrix/matrix.h"
#include "../matrix/operations.h"

NeuralNetwork* nn_create(int num_inputs, int num_hiddens, int num_outputs, double learning_rate) {
    NeuralNetwork* nn = malloc(sizeof(NeuralNetwork));
    if(!nn) return NULL;

    nn->num_inputs = num_inputs;
    nn->num_hiddens = num_hiddens;
    nn->num_outputs = num_outputs;
    nn->learning_rate = learning_rate;

    nn->W1 = matrix_create(num_hiddens, num_inputs);
    nn->b1 = matrix_create(num_hiddens, 1);
    nn->W2 = matrix_create(num_outputs, num_hiddens);
    nn->b2 = matrix_create(num_outputs, 1);

    matrix_randomize(nn->W1, num_inputs);
    matrix_init(nn->b1, 0.0);
    matrix_randomize(nn->W2, num_hiddens);
    matrix_init(nn->b2, 0.0);

    nn->hidden_weights = hidden_layer;
    nn->output_weights = output_layer;
    return nn;
}

void nn_free(NeuralNetwork* nn) {
    if(!nn) return NULL;

    matrix_init(nn->W1, 0);
    matrix_init(nn->b1, 0);
    matrix_init(nn->W2, 0);
    matrix_init(nn->b2, 0);
    free(nn);
}

void forward(NeuralNetwork* nn, Matrix* x, Matrix** z1, Matrix** a1, Matrix** z2, Matrix** a2) {
    Matrix* W1x = dot(nn->W1, x);
    *z1 = add(W1x, nn->b1);
    matrix_free(W1x);

    *a1 = apply(sigmoid, *z1);

    Matrix* W2a1 = dot(nn->W2, *a1);
    *z2 = add(W2a1, nn->b2);

    *a2 = softmax(*z2);
}