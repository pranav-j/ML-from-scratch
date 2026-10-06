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

SRCS   = main.c lstm/lstm.c
OBJS   = $(SRCS:.c=.o)
DEPS   = $(OBJS:.o=.d)
TARGET = lstm_

# --- Gradient checker: separate binary, run once to verify backward ---   # NEW
GC_SRCS = gradient_check.c lstm/lstm.c                                     # NEW
GC_OBJS = $(GC_SRCS:.c=.o)                                                 # NEW
GC_DEPS = $(GC_OBJS:.o=.d)                                                 # NEW
GC_BIN  = gradient_check                                                   # NEW

all: $(TARGET)

$(TARGET): $(OBJS) $(LIBS)
	$(CC) $(OBJS) $(LIBS) $(LDLIBS) -o $@

$(GC_BIN): $(GC_OBJS) $(LIBS)                                              # NEW
	$(CC) $(GC_OBJS) $(LIBS) $(LDLIBS) -o $@                               # NEW

# --- Debug build ---
# Convenience target: build with sanitizers.
debug: CFLAGS += -O0 -g -fsanitize=address,undefined
debug: LDLIBS += -fsanitize=address,undefined
debug: clean $(TARGET)

# --- Gradient check: always sanitized, always rebuilt from scratch ---     # NEW
gradcheck: CFLAGS += -O0 -g -fsanitize=address,undefined                   # NEW
gradcheck: LDLIBS += -fsanitize=address,undefined                          # NEW
gradcheck: clean $(GC_BIN)                                                 # NEW

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

clean:                                                                     # CHANGED
	rm -f $(OBJS) $(GC_OBJS) $(DEPS) $(GC_DEPS) $(TARGET) $(GC_BIN)

distclean: clean
	$(MAKE) -C $(LIBDIR)/matrix clean
	$(MAKE) -C $(LIBDIR)/activations clean
	$(MAKE) -C $(LIBDIR)/data_ops clean

-include $(DEPS) $(GC_DEPS)                                                # CHANGED

.PHONY: all clean distclean debug gradcheck                                # CHANGED