CC = gcc

CFLAGS  = -std=c11 -fopenmp -O2 -ggdb -pedantic -Wall -Wextra
CFLAGS += -Wno-missing-field-initializers -Wno-unused-parameter -Wold-style-definition -Wdeclaration-after-statement -Wmissing-declarations -Wmissing-prototypes -Wredundant-decls -Wmissing-noreturn -Wshadow -Wpointer-arith -Wcast-align -Wwrite-strings -Winline -Wformat-nonliteral -Wformat-security -Wswitch-enum -Wswitch-default -Winit-self -Wmissing-include-dirs -Wundef -Waggregate-return -Wmissing-format-attribute -Wnested-externs -Wstrict-prototypes
LDLIBS = -lm

PROGRAMS = checkpoint

all: $(PROGRAMS)

clean:
	rm -f $(PROGRAMS)
