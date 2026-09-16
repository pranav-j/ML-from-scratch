#include <matrix/matrix.h>
#include <matrix/operations.h>
#include "data_ops/data.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

Matrix* softmax(Matrix* matrix) {
    double total = 0;
    for(int i = 0; i < matrix->rows; i++) {
        for(int j = 0; j < matrix->cols; j++) {
            total += exp(matrix->values[i][j]);
        }
    }

    Matrix* out = matrix_create(matrix->rows, matrix->cols);
    if(!out) return NULL;

    for(int i = 0; i < matrix->rows; i++) {
        for(int j = 0; j < matrix->cols; j++) {
            out->values[i][j] = exp(matrix->values[i][j])/total;
        }
    }

    return out;
}    

Matrix* forward(Matrix* Wxh, Matrix* Whh, Matrix* Why, Matrix* bh, Matrix* by, Matrix* x) {
    double total_loss;
    for(int i = 0; i < T; i++) {
        Matrix* char = matrix_create(c->vocab_size, 1);
        Matrix* next_char = matrix_create(c->vocab_size, 1);

        for(int j = 0; j < c->vocab_size; j++) {
            char[j][1] = x[j][i];
            next_char[j][1] = x[j][i + 1]; 
        }
        Matrix* a = dot(Whx, char) + dot(Whh, h) + bh;
        Matrix* h = apply(tanh, a);
        Matrix* z = dot(Why, h) + by;
        y = softmax(z);
        double loss = -dot(next_char, log(char));
        total_loss += loss;
    }
    
}

int main(void) {
    Corpus* c = corpus_load("data/corpus.txt");
    printf("Text length %d \n", c->text_length);
    printf("Vocab size %d \n", c->vocab_size);

    int H = 100;
    int T = 25;

    Matrix* Wxh = matrix_create(c->vocab_size, H);
    Matrix* Whh = matrix_create(c->vocab_size, H);
    Matrix* bh = matrix_create(H, 1);
    Matrix* Why = matrix_create(H, c->vocab_size);
    Matrix* by = matrix_create(H, 1);

    matrix_randomize(Wxh, H);
    matrix_randomize(Whh, H);
    matrix_randomize(Why, H);



    corpus_free(c);
    return 0;
}
