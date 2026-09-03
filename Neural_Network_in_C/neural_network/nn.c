#include <math.h>
#include <stdio.h>
#include <stdlib.h>

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

double nn_train_one(NeuralNetwork* nn, Matrix* x, Matrix* y) {
    Matrix *z1, *a1, *z2, *a2;

    // FORWARD
    forward(nn, x, &z1, &a1, &z2, &a2);

    // LOSS
    double loss = 0.0;
    for(int i = 0; i < y->rows; i++) {
        loss += -y->values[i][0] * log(a2->values[1][0]);
    }


    // BACKWARD
    Matrix* dz2 = subtract(a2, y);

    Matrix* dW2 = dot(dz2, transpose(a1));

    Matrix* db2 = dz2;

    Matrix* da1 = dot(transpose(nn->W2), dz2);
    
    Matrix* sp = apply(sigmoidPrime, a1);
    Matrix* dz1 = hadamard(da1, sp);

    Matrix* dW1 = dot(dz1, transpose(x));

    Matrix* db1 = dz1;

    // UPDATE
    matrix_update(nn->W2, dW2, nn->learning_rate);
    matrix_update(nn->b2, db2, nn->learning_rate);
    matrix_update(nn->W1, dW1, nn->learning_rate);
    matrix_update(nn->b1, db1, nn->learning_rate);

    // FREE INTERMEDIARIES
    matrix_free(z1); matrix_free(a1); matrix_free(z2); matrix_free(a2);
    matrix_free(dz2);   // this also disposes of db2 (same pointer)
    matrix_free(a1_T);
    matrix_free(dW2);
    matrix_free(W2_T);
    matrix_free(da1);
    matrix_free(sp);
    matrix_free(dz1);   // this also disposes of db1 (same pointer)
    matrix_free(x_T);
    matrix_free(dW1);

    return loss;
}