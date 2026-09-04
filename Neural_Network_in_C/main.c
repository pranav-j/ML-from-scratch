#include "matrix/matrix.h"
#include "neural_network/nn.h"
#include "mnist_ops/mnist.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static void shuffle(int* idx, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int t = idx[i]; idx[i] = idx[j]; idx[j] = t;
    }
}

static double evaluate(NeuralNetwork* nn, MnistData* data) {
    int correct = 0;
    for (int i = 0; i < data->count; i++) {
        int pred = nn_predict_class(nn, data->samples[i].input);
        if (pred == data->samples[i].label) correct++;
    }
    return (double)correct / data->count;
}

int main(int argc, char** argv) {
    const char* train_csv = (argc > 1) ? argv[1] : "data/mnist_train.csv";
    const char* test_csv  = (argc > 2) ? argv[2] : "data/mnist_test.csv";
    int train_max = (argc > 3) ? atoi(argv[3]) : 10000;
    int test_max  = (argc > 4) ? atoi(argv[4]) : 1000;
    int epochs    = (argc > 5) ? atoi(argv[5]) : 3;

    srand((unsigned)time(NULL));

    printf("Loading training data from %s (up to %d samples)...\n", train_csv, train_max);
    MnistData* train = mnist_load(train_csv, train_max);
    if (!train) { fprintf(stderr, "Failed to load training data\n"); return 1; }
    printf("Loaded %d training samples.\n", train->count);

    printf("Loading test data from %s (up to %d samples)...\n", test_csv, test_max);
    MnistData* test = mnist_load(test_csv, test_max);
    if (!test) { fprintf(stderr, "Failed to load test data\n"); return 1; }
    printf("Loaded %d test samples.\n", test->count);

    // 784 -> 300 -> 10, sigmoid + softmax, lr = 0.1
    NeuralNetwork* nn = nn_create(784, 300, 10, 0.1);
    if (!nn) { fprintf(stderr, "Failed to create network\n"); return 1; }

    printf("\nInitial test accuracy: %.2f%%\n", evaluate(nn, test) * 100.0);

    int* idx = malloc(train->count * sizeof(int));
    for (int i = 0; i < train->count; i++) idx[i] = i;

    for (int epoch = 0; epoch < epochs; epoch++) {
        shuffle(idx, train->count);
        double total_loss = 0.0;
        for (int i = 0; i < train->count; i++) {
            int k = idx[i];
            total_loss += nn_train_one(nn, train->samples[k].input,
                                            train->samples[k].target);
            if ((i + 1) % 1000 == 0) {
                printf("\r  epoch %d  [%d/%d]  avg_loss = %.4f",
                       epoch + 1, i + 1, train->count, total_loss / (i + 1));
                fflush(stdout);
            }
        }
        double acc = evaluate(nn, test);
        printf("\n  epoch %d done. avg_loss=%.4f  test_acc=%.2f%%\n",
               epoch + 1, total_loss / train->count, acc * 100.0);
    }

    free(idx);
    nn_free(nn);
    mnist_free(train);
    mnist_free(test);
    return 0;
}