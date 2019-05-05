CFLAGS  = -std=c11 -O2 -ggdb -Wpedantic -Wall -Wextra -Waggregate-return -Wcast-align -Wcast-qual -Wdeclaration-after-statement -Wdouble-promotion -Wduplicated-cond -Wfloat-equal -Wformat=2 -Winit-self -Winline -Wjump-misses-init -Wlogical-op -Wmissing-declarations -Wmissing-format-attribute -Wmissing-include-dirs -Wmissing-noreturn -Wmissing-prototypes -Wnested-externs -Wnull-dereference -Wold-style-definition -Wredundant-decls -Wrestrict -Wshadow -Wstrict-prototypes -Wswitch-default -Wswitch-enum -Wundef -Wuninitialized -Wwrite-strings -Wno-unused-parameter
LDLIBS =

CFLAGS += $(shell pkg-config --cflags fuse)
LDLIBS += $(shell pkg-config --libs fuse)

PROGRAMS = dummyfs

all: $(PROGRAMS)

clean:
	rm -f $(PROGRAMS)
