#include "../matrix/matrix.h"

typedef struct {
    int num_inputs;
    int num_hiddens;
    int num_outputs;
    double learning_rate;
    Matrix* hidden_weights;
    Matrix* output_weights;
} NeuraNetwork;