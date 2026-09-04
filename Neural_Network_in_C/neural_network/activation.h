#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "../matrix/matrix.h"

double sigmoid(double x);
double sigmoid_prime_from_a(double a);
Matrix* softmax(Matrix* matrix);

#endif