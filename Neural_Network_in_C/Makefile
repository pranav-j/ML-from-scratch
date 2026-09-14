CC       = cc
LIBDIR   = ../lib
MATRIX   = $(LIBDIR)/matrix/libmatrix.a
CFLAGS   = -O2 -Wall -Wextra -I$(LIBDIR)
LDLIBS   = -lm

SRCS     = main.c neural_network/nn.c neural_network/activation.c mnist_ops/mnist.c
OBJS     = $(SRCS:.c=.o)
TARGET   = nn

all: $(TARGET)

$(TARGET): $(OBJS) $(MATRIX)
	$(CC) $(OBJS) $(MATRIX) $(LDLIBS) -o $@

# Delegate to the shared library's own Makefile so it is always up to date.
$(MATRIX): $(wildcard $(LIBDIR)/matrix/*.c) $(wildcard $(LIBDIR)/matrix/*.h)
	$(MAKE) -C $(LIBDIR)/matrix

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

# Also cleans the shared library.
distclean: clean
	$(MAKE) -C $(LIBDIR)/matrix clean

.PHONY: all clean distclean
