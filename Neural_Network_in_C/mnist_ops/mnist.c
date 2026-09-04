#include "mnist.h"
#include "../matrix/matrix.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Big enough for one CSV line: 785 numbers × ~5 chars + commas ≈ 4KB.
#define LINE_BUF 8192

static int parse_line(char* line, int* label, double* pixels) {
    // Parse "label,p0,p1,...,p783\n" using strtol. Returns 1 on success, 0 on failure.
    char* p = line;
    char* end;

    long v = strtol(p, &end, 10);
    if (end == p) return 0;
    *label = (int)v;
    p = end;

    for (int i = 0; i < 784; i++) {
        if (*p != ',') return 0;    // expect a comma before each pixel
        p++;                        // skip it
        v = strtol(p, &end, 10);
        if (end == p) return 0;
        pixels[i] = v / 255.0;      // normalize to [0, 1]
        p = end;
    }
    return 1;
}

MnistData* mnist_load(const char* path, int max_samples) {
    FILE* file = fopen(path, "r");
    if (!file) { perror(path); return NULL; }

    char line[LINE_BUF];

    // Skip header row
    if (!fgets(line, LINE_BUF, file)) { fclose(file); return NULL; }

    // If max_samples <= 0 we'll grow dynamically; start with a small capacity.
    int cap = (max_samples > 0) ? max_samples : 1024;
    MnistSample* samples = malloc(cap * sizeof(MnistSample));
    if (!samples) { fclose(file); return NULL; }

    double pixels[784];
    int count = 0;

    while (fgets(line, LINE_BUF, file)) {
        if (max_samples > 0 && count >= max_samples) break;

        // Grow if needed (only when max_samples <= 0)
        if (count >= cap) {
            int new_cap = cap * 2;
            MnistSample* g = realloc(samples, new_cap * sizeof(MnistSample));
            if (!g) break;
            samples = g;
            cap = new_cap;
        }

        int label;
        if (!parse_line(line, &label, pixels)) continue;   // skip malformed lines

        Matrix* input  = matrix_create(784, 1);
        Matrix* target = matrix_create(10, 1);
        if (!input || !target) {
            matrix_free(input); matrix_free(target);
            break;
        }
        for (int i = 0; i < 784; i++) input->values[i][0] = pixels[i];
        for (int i = 0; i < 10;  i++) target->values[i][0] = 0.0;
        target->values[label][0] = 1.0;

        samples[count].input  = input;
        samples[count].target = target;
        samples[count].label  = label;
        count++;
    }

    fclose(file);

    MnistData* data = malloc(sizeof(MnistData));
    data->samples = samples;
    data->count   = count;
    return data;
}

void mnist_free(MnistData* data) {
    if (!data) return;
    for (int i = 0; i < data->count; i++) {
        matrix_free(data->samples[i].input);
        matrix_free(data->samples[i].target);
    }
    free(data->samples);
    free(data);
}