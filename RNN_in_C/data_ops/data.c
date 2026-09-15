#include <matrix/matrix.h>
#include "data.h"

#include <stdlib.h>
#include <stdio.h>

Corpus* corpus_load(const char* file_name) {
    FILE* f = fopen(file_name, "rb");
    if(!f) {
        fprintf(stderr, "corpus_load: cannot open %s\n", filename);
        return NULL;
    }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    rewind(f);

    Corpus* c = malloc(sizeof(Corpus));
    c->text = malloc(size + 1);
    c->text_length = (int)fread(c->text, 1, size, f);
    c->text[text_length + 1] = '\0';
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
