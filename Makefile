CXX = g++
CXXFLAGS = -std=c++20 -Wall -MMD -g

SRC = ./cpp

SRCS = $(shell find $(SRC) -name '*.cpp')

OBJS = ${SRCS:.cpp=.o}

DPDS = ${OBJS:.o=.d}

all: pipe
pipe: $(OBJS)
	$(CXX) $(OBJS) -o pipe

-include ${DPDS}

.PHONY: clean

clean:
	rm ${OBJS} ${DPDS}
