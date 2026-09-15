#include <matrix/matrix.h>
#include <matrix/operations.h>
#include "data_ops/data.h"

#include <stdio.h>
#include <stdlib.h>

int main(void) {
    Corpus* c = corpus_load("data/corpus.txt");
    printf("Text length %d \n", c->text_length);
    printf("Vocab size %d \n", c->vocab_size);

    corpus_free(c);
    return 0;
}
