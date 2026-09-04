#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "../matrix/matrix.h"

double sigmoid(double x);
Matrix* sigmoidPrime(Matrix* input);
Matrix* softmax(Matrix* matrix);

#endif