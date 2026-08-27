#include "../matrix/matrix.h"
#include "../matrix/operations.h"
#include "activation.h"

NeuralNetwork* create_nn(int num_inputs, int num_hiddens, int num_outputs, double learning_rate) {
    NeuralNetwork* nn = malloc(sizeof(NeuralNetwork));
    nn->num_inputs = num_inputs;
    nn->num_hiddens = num_hiddens;
    nn->num_outputs = num_outputs;
    Matrix* hidden_layer = matrix_create(num_hiddens, num_inputs);
    Matrix* output_layer = matrix_create(num_outputs, num_hiddens);
    matrix_randomize(hidden_layer, num_hiddens);
    matrix_randomize(output_layer, num_outputs);
    nn->hidden_weights = hidden_layer;
    nn->output_weights = output_layer;
    return nn;
}