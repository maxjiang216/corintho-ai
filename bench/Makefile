# Benchmark harness for the Corintho engine.
#
# Deliberately separate from the top-level CMakeLists.txt, which forces
# --coverage -pg into every build and so cannot produce a clean timing binary.
#
#   make            build all benchmarks
#   make micro      game-layer microbenchmarks
#   make selfplay   end-to-end engine benchmark (stub evaluator)
#   make golden     behavioural digests (see golden.cpp)
#   make run        build and run everything at default sizes
#   make callgrind  instruction-level attribution
#
# For the before/after workflow use ./run_suite.sh, not these targets
# directly -- see README.md.

CXX      ?= g++
SRC_DIR  := ../corintho_ai/cpp/src
INC      := -I../corintho_ai/cpp/include -I../gsl/include -I.
CXXFLAGS := -std=c++17 -O3 -DNDEBUG -g -flto -march=native -fopenmp -Wall -Wextra
LDFLAGS  := -fopenmp -flto -O3 -march=native

BUILD := build
CORE  := $(BUILD)/util.o $(BUILD)/move.o $(BUILD)/game.o $(BUILD)/node.o \
         $(BUILD)/trainmc.o $(BUILD)/selfplayer.o $(BUILD)/trainer.o

.PHONY: all micro selfplay golden run callgrind clean

all: micro selfplay golden

micro: $(BUILD)/micro_bench
selfplay: $(BUILD)/selfplay_bench
golden: $(BUILD)/golden

$(BUILD):
	@mkdir -p $(BUILD)

$(BUILD)/%.o: $(SRC_DIR)/%.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@

$(BUILD)/micro.o: micro.cpp bench_util.h | $(BUILD)
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@

$(BUILD)/selfplay.o: selfplay.cpp bench_util.h | $(BUILD)
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@

$(BUILD)/golden.o: golden.cpp bench_util.h | $(BUILD)
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@

$(BUILD)/micro_bench: $(BUILD)/micro.o $(CORE)
	$(CXX) $^ $(LDFLAGS) -o $@

$(BUILD)/selfplay_bench: $(BUILD)/selfplay.o $(CORE)
	$(CXX) $^ $(LDFLAGS) -o $@

$(BUILD)/golden: $(BUILD)/golden.o $(CORE)
	$(CXX) $^ $(LDFLAGS) -o $@

run: all
	@./$(BUILD)/golden
	@./$(BUILD)/micro_bench
	@./$(BUILD)/selfplay_bench

# perf is unusable here (perf_event_paranoid=4 needs root), so callgrind is the
# supported profiler. Instructions, not cycles, single-threaded -- see README.md.
callgrind: selfplay
	valgrind --tool=callgrind --callgrind-out-file=cg.out \
	  ./$(BUILD)/selfplay_bench 3 400 16 1 > /dev/null 2>&1
	callgrind_annotate --auto=no cg.out | head -40

clean:
	rm -rf $(BUILD) perf.data cg.out
