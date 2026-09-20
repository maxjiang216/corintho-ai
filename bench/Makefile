# Benchmark harness for the Corintho engine.
#
# Deliberately separate from the top-level CMakeLists.txt, which forces
# --coverage -pg into every build and so cannot produce a clean timing binary.
#
#   make            build both benchmarks
#   make micro      game-layer microbenchmarks
#   make selfplay   end-to-end engine benchmark (stub evaluator)
#   make run        build and run both at default sizes
#   make perf       record a profile of the engine benchmark

CXX      ?= g++
SRC_DIR  := ../corintho_ai/cpp/src
INC      := -I../corintho_ai/cpp/include -I../gsl/include -I.
CXXFLAGS := -std=c++17 -O3 -DNDEBUG -flto -march=native -fopenmp -Wall -Wextra
LDFLAGS  := -fopenmp -flto -O3 -march=native

BUILD := build
CORE  := $(BUILD)/util.o $(BUILD)/move.o $(BUILD)/game.o $(BUILD)/node.o \
         $(BUILD)/trainmc.o $(BUILD)/selfplayer.o $(BUILD)/trainer.o

.PHONY: all micro selfplay run perf clean

all: micro selfplay

micro: $(BUILD)/micro_bench
selfplay: $(BUILD)/selfplay_bench

$(BUILD):
	@mkdir -p $(BUILD)

$(BUILD)/%.o: $(SRC_DIR)/%.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@

$(BUILD)/micro.o: micro.cpp bench_util.h | $(BUILD)
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@

$(BUILD)/selfplay.o: selfplay.cpp bench_util.h | $(BUILD)
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@

$(BUILD)/micro_bench: $(BUILD)/micro.o $(CORE)
	$(CXX) $^ $(LDFLAGS) -o $@

$(BUILD)/selfplay_bench: $(BUILD)/selfplay.o $(CORE)
	$(CXX) $^ $(LDFLAGS) -o $@

run: all
	@./$(BUILD)/micro_bench
	@./$(BUILD)/selfplay_bench

perf: selfplay
	perf record -g --call-graph=dwarf -o perf.data ./$(BUILD)/selfplay_bench 200 800 16 14
	perf report -i perf.data --stdio | head -50

clean:
	rm -rf $(BUILD) perf.data
