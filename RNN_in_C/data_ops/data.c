#include <matrix/matrix.h>
#include "data.h"

#include <stdlib.h>
#include <stdio.h>

Corpus* corpus_load(const char* file_name) {
    FILE* f = fopen(file_name, "rb");
    if(!f) {
        fprintf(stderr, "corpus_load: cannot open %s\n", file_name);
        return NULL;
    }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    rewind(f);

    Corpus* c = malloc(sizeof(Corpus));
    c->text = malloc(size + 1);
    c->text_length = (int)fread(c->text, 1, size, f);
    c->text[c->text_length] = '\0';
    fclose(f);

    for(int i = 0; i <= 255; i++) {
        c->index_to_char[i] = 0;
        c->char_to_index[i] = -1;
    }

    c->vocab_size = 0;

    for(int i = 0; i < c->text_length; i++) {
        unsigned char ch = (unsigned char) c->text[i];
        if(c->char_to_index[ch] == -1) {
            c->index_to_char[c->vocab_size] = (char) ch;
            c->char_to_index[ch] = c->vocab_size;
            c->vocab_size++;
        }
    }

    return c;
}

void corpus_free(Corpus* c) {
    if(!c) return;
    free(c->text);
    free(c);   
}

Matrix* one_hot(int index, int vocab_size) {
    Matrix* m = matrix_create(vocab_size, 1);
    matrix_init(m, 0.0);
    m->values[index][0] = 1.0;
    return m;
}