CC       = cc
LIBDIR   = ../lib
MATRIX   = $(LIBDIR)/matrix/libmatrix.a
# -MMD -MP makes the compiler emit a .d file recording which headers each
# source pulled in, so editing a header rebuilds everything that includes it.
CFLAGS   = -O2 -Wall -Wextra -I$(LIBDIR) -MMD -MP
LDLIBS   = -lm

# Add new sources here as the RNN grows.
SRCS     = main.c data_ops/data.c rnn/rnn.c
OBJS     = $(SRCS:.c=.o)
DEPS     = $(OBJS:.o=.d)
TARGET   = rnn_

all: $(TARGET)

$(TARGET): $(OBJS) $(MATRIX)
	$(CC) $(OBJS) $(MATRIX) $(LDLIBS) -o $@

# Delegate to the shared library's own Makefile so it is always up to date.
$(MATRIX): $(wildcard $(LIBDIR)/matrix/*.c) $(wildcard $(LIBDIR)/matrix/*.h)
	$(MAKE) -C $(LIBDIR)/matrix

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(DEPS) $(TARGET)

# Also cleans the shared library.
distclean: clean
	$(MAKE) -C $(LIBDIR)/matrix clean

# Pull in the generated header dependencies. The leading '-' means "don't
# complain if they don't exist yet" (they won't, on a clean tree).
-include $(DEPS)

.PHONY: all clean distclean
