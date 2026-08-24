#include "activation.h"
#include "../matrix/matrix.h"
#include "../matrix/operations.h"

#include <math.h>

double sigmoid(double x) {
    return 1.0 / (1 + exp(-x));
}

