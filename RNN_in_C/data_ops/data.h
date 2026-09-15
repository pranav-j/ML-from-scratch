#include <matrix/matrix.h>

typedef struct {
    char*  text;
    int text_length;
    int vocab_size;
    int index_to_char[256];
    char char_to_index[256];
} Corpus;

Corpus* corpus_load(const char* file_name);
void corpus_free(Corpus* c);
Matrix* one_hot(int index, int vocab_size); 