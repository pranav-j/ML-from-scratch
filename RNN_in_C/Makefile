CC       = cc
LIBDIR   = ../lib
MATRIX   = $(LIBDIR)/matrix/libmatrix.a
ACT      = $(LIBDIR)/activations/libactivations.a
DATA     = $(LIBDIR)/data_ops/libdata.a
LIBS     = $(MATRIX) $(DATA) $(ACT)
# -MMD -MP makes the compiler emit a .d file recording which headers each
# source pulled in, so editing a header rebuilds everything that includes it.
CFLAGS   = -O2 -Wall -Wextra -Werror -I$(LIBDIR) -MMD -MP
LDLIBS   = -lm

# --- Shared objects (used by both targets) ---
SHARED_SRCS = rnn/rnn.c
SHARED_OBJS = $(SHARED_SRCS:.c=.o)

# --- Main training binary ---
SRCS   = main.c $(SHARED_SRCS)
OBJS   = $(SRCS:.c=.o)
TARGET = rnn_

# --- Gradient check binary ---
GC_SRCS   = gradient_check.c $(SHARED_SRCS)
GC_OBJS   = $(GC_SRCS:.c=.o)
GC_TARGET = gradcheck

DEPS = $(OBJS:.o=.d) $(GC_OBJS:.o=.d)

all: $(TARGET)

$(TARGET): $(OBJS) $(LIBS)
	$(CC) $(OBJS) $(LIBS) $(LDLIBS) -o $@

$(GC_TARGET): $(GC_OBJS) $(LIBS)
	$(CC) $(GC_OBJS) $(LIBS) $(LDLIBS) -o $@

# --- Debug builds ---
# Convenience target: build training binary with sanitizers.
debug: CFLAGS += -O0 -g -fsanitize=address,undefined
debug: LDLIBS += -fsanitize=address,undefined
debug: clean $(TARGET)

# Convenience target: build gradcheck binary with sanitizers.
# Named "gc-debug" (not "gradcheck") to avoid colliding with the binary rule above.
gc-debug: CFLAGS += -O0 -g -fsanitize=address,undefined
gc-debug: LDLIBS += -fsanitize=address,undefined
gc-debug: clean $(GC_TARGET)

# --- Libraries ---
$(MATRIX): $(wildcard $(LIBDIR)/matrix/*.c) $(wildcard $(LIBDIR)/matrix/*.h)
	$(MAKE) -C $(LIBDIR)/matrix

$(ACT): $(wildcard $(LIBDIR)/activations/*.c) $(wildcard $(LIBDIR)/activations/*.h)
	$(MAKE) -C $(LIBDIR)/activations

$(DATA): $(wildcard $(LIBDIR)/data_ops/*.c) $(wildcard $(LIBDIR)/data_ops/*.h)
	$(MAKE) -C $(LIBDIR)/data_ops

# --- Compilation rule ---
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(GC_OBJS) $(DEPS) $(TARGET) $(GC_TARGET)

distclean: clean
	$(MAKE) -C $(LIBDIR)/matrix clean
	$(MAKE) -C $(LIBDIR)/activations clean
	$(MAKE) -C $(LIBDIR)/data_ops clean

-include $(DEPS)

.PHONY: all clean distclean debug gc-debug