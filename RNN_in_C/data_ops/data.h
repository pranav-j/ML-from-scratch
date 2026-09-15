#ifndef DATA_H
#define DATA_H

#include <matrix/matrix.h>

typedef struct {
    char*  text;
    int text_length;
    int vocab_size;
    char index_to_char[256];
    int char_to_index[256];
} Corpus;

Corpus* corpus_load(const char* file_name);
void corpus_free(Corpus* c);
Matrix* one_hot(int index, int vocab_size);

#endif