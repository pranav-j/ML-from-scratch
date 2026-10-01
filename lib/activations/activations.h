#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <matrix/matrix.h>

double sigmoid(double x);

// Derivatives expressed in terms of the activation's own output, which is what
// backprop has on hand: sigma'(z) = a(1-a) and tanh'(z) = 1 - a^2.
double sigmoid_prime_from_a(double a);
double tanh_prime_from_a(double a);

// Numerically stable softmax over every element of the matrix (they form one
// distribution). Subtracting the max before exp() keeps large logits from
// overflowing; it cancels out in the ratio.
Matrix* softmax(Matrix* matrix);

#endif
