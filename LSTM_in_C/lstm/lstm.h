#include <matrix/matrix.h>

typedef struct {
    int H;
    int V;

    Matrix* W;
    Matrix* b;

    Matrix* Wy;
    Matrix* by;
} LSTM;