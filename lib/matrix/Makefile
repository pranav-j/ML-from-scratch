# Shared matrix library. Built as a static lib and linked by each project.
# Include it from a consumer with:  #include <matrix/matrix.h>   (compile with -I<repo>/lib)

CC      = cc
CFLAGS  = -O2 -Wall -Wextra -I..

SRCS    = matrix.c operations.c
OBJS    = $(SRCS:.c=.o)
TARGET  = libmatrix.a

all: $(TARGET)

$(TARGET): $(OBJS)
	ar rcs $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
