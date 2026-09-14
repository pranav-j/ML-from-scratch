# Shared matrix library. Built as a static lib and linked by each project.
# Include it from a consumer with:  #include <matrix/matrix.h>   (compile with -I<repo>/lib)

CC      = cc
# -MMD -MP makes the compiler emit a .d file recording which headers each
# source pulled in, so editing a header rebuilds everything that includes it.
CFLAGS  = -O2 -Wall -Wextra -I.. -MMD -MP

SRCS    = matrix.c operations.c
OBJS    = $(SRCS:.c=.o)
DEPS    = $(OBJS:.o=.d)
TARGET  = libmatrix.a

all: $(TARGET)

$(TARGET): $(OBJS)
	ar rcs $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(DEPS) $(TARGET)

# Pull in the generated header dependencies. The leading '-' means "don't
# complain if they don't exist yet" (they won't, on a clean tree).
-include $(DEPS)

.PHONY: all clean
