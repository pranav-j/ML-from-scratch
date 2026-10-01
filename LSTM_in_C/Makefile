CC       = cc
LIBDIR   = ../lib
MATRIX   = $(LIBDIR)/matrix/libmatrix.a
DATA     = $(LIBDIR)/data_ops/libdata.a
LIBS     = $(MATRIX) $(DATA)
# -MMD -MP makes the compiler emit a .d file recording which headers each
# source pulled in, so editing a header rebuilds everything that includes it.
CFLAGS   = -O2 -Wall -Wextra -Werror -I$(LIBDIR) -MMD -MP
LDLIBS   = -lm

SRCS   = main.c lstm/lstm.c
OBJS   = $(SRCS:.c=.o)
DEPS   = $(OBJS:.o=.d)
TARGET = lstm_

all: $(TARGET)

$(TARGET): $(OBJS) $(LIBS)
	$(CC) $(OBJS) $(LIBS) $(LDLIBS) -o $@

# --- Debug build ---
# Convenience target: build with sanitizers.
debug: CFLAGS += -O0 -g -fsanitize=address,undefined
debug: LDLIBS += -fsanitize=address,undefined
debug: clean $(TARGET)

# --- Libraries ---
$(MATRIX): $(wildcard $(LIBDIR)/matrix/*.c) $(wildcard $(LIBDIR)/matrix/*.h)
	$(MAKE) -C $(LIBDIR)/matrix

$(DATA): $(wildcard $(LIBDIR)/data_ops/*.c) $(wildcard $(LIBDIR)/data_ops/*.h)
	$(MAKE) -C $(LIBDIR)/data_ops

# --- Compilation rule ---
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(DEPS) $(TARGET)

distclean: clean
	$(MAKE) -C $(LIBDIR)/matrix clean
	$(MAKE) -C $(LIBDIR)/data_ops clean

-include $(DEPS)

.PHONY: all clean distclean debug
