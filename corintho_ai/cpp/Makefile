CXX = g++
CXXFLAGS = -std=c++20 -Wall -MMD -O3 -fopenmp -g #-pg

SRC = .

SRCS = $(shell find $(SRC) -name '*.cpp')

OBJS = ${SRCS:.cpp=.o}

DPDS = ${OBJS:.o=.d}

all: pipe
pipe: $(CXXFLGS) $(OBJS)
	$(CXX)  $(OBJS) $(CXXFLGS) -O3 -fopenmp -g -o pipe

-include ${DPDS}

.PHONY: clean

clean:
	rm ${OBJS} ${DPDS}
