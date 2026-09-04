#ifndef NN_H
#define NN_H

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
} NeuralNetwork;

NeuralNetwork* nn_create(int num_inputs, int num_hiddens, int num_outputs, double learning_rate);
void nn_free(NeuralNetwork* nn);
void forward(NeuralNetwork* nn, Matrix* x, Matrix** z1, Matrix** a1, Matrix** z2, Matrix** a2);
double nn_train_one(NeuralNetwork* nn, Matrix* x, Matrix* y);
Matrix* nn_predict(NeuralNetwork* nn, Matrix* x);
int nn_predict_class(NeuralNetwork* nn, Matrix* x);

#endif