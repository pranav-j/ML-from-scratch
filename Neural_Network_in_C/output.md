pranav-j@Pranavs-Air Neural_Network_in_C % clang -std=c23 -Wall -Wextra -g -fsanitize=address,undefined \
    matrix/matrix.c matrix/operations.c \
    neural_network/activation.c neural_network/nn.c mnist_ops/mnist.c \
    main.c \
    -o nn -lm
pranav-j@Pranavs-Air Neural_Network_in_C % ./nn                                                          
Loading training data from data/mnist_train.csv (up to 10000 samples)...
Loaded 10000 training samples.
Loading test data from data/mnist_test.csv (up to 1000 samples)...
Loaded 1000 test samples.

Initial test accuracy: 10.70%
  epoch 1  [10000/10000]  avg_loss = 8.8153
  epoch 1 done. avg_loss=8.8153  test_acc=89.80%
  epoch 2  [10000/10000]  avg_loss = 10.5490
  epoch 2 done. avg_loss=10.5490  test_acc=91.70%
  epoch 3  [10000/10000]  avg_loss = 11.3557
  epoch 3 done. avg_loss=11.3557  test_acc=93.50%
pranav-j@Pranavs-Air Neural_Network_in_C % ./nn data/mnist_train.csv data/mnist_test.csv 60000 10000 10
Loading training data from data/mnist_train.csv (up to 60000 samples)...
Loaded 60000 training samples.
Loading test data from data/mnist_test.csv (up to 10000 samples)...
Loaded 10000 test samples.

Initial test accuracy: 9.80%
  epoch 1  [60000/60000]  avg_loss = 9.9799
  epoch 1 done. avg_loss=9.9799  test_acc=96.16%
  epoch 2  [60000/60000]  avg_loss = 11.5364
  epoch 2 done. avg_loss=11.5364  test_acc=96.85%
  epoch 3  [60000/60000]  avg_loss = 12.4730
  epoch 3 done. avg_loss=12.4730  test_acc=97.20%
  epoch 4  [60000/60000]  avg_loss = 13.6521
  epoch 4 done. avg_loss=13.6521  test_acc=97.53%
  epoch 5  [60000/60000]  avg_loss = 14.6134
  epoch 5 done. avg_loss=14.6134  test_acc=97.75%
  epoch 6  [60000/60000]  avg_loss = 15.3550
  epoch 6 done. avg_loss=15.3550  test_acc=97.69%
  epoch 7  [60000/60000]  avg_loss = 16.3192
  epoch 7 done. avg_loss=16.3192  test_acc=97.98%
  epoch 8  [60000/60000]  avg_loss = 17.0406
  epoch 8 done. avg_loss=17.0406  test_acc=98.13%
  epoch 9  [60000/60000]  avg_loss = 17.9563
  epoch 9 done. avg_loss=17.9563  test_acc=98.11%
  epoch 10  [60000/60000]  avg_loss = 18.6007
  epoch 10 done. avg_loss=18.6007  test_acc=98.18%