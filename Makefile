CXX = g++
CXXFLAGS = -std=c++20 -Wall -MMD -O3 -g #-pg

SRC = ./cpp

SRCS = $(shell find $(SRC) -name '*.cpp')

OBJS = ${SRCS:.cpp=.o}

DPDS = ${OBJS:.o=.d}

all: pipe
pipe: $(CXXFLGS) $(OBJS)
	$(CXX) $(CXXFLGS) $(OBJS) -O3 -o pipe

-include ${DPDS}

.PHONY: clean

clean:
	rm ${OBJS} ${DPDS}
