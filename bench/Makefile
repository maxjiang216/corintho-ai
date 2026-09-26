# Benchmark harness for the Corintho engine.
#
# Deliberately separate from the top-level CMakeLists.txt, which forces
# --coverage -pg into every build and so cannot produce a clean timing binary.
#
#   make            build all benchmarks
#   make micro      game-layer microbenchmarks
#   make selfplay   end-to-end engine benchmark (stub evaluator)
#   make mlp_bench  throughput of the in-process CPU network
#   make golden     behavioural digests (see golden.cpp)
#   make verify     differential test against the frozen reference
#   make rulecheck  exhaustive line-breaking test (fails until item 4 is fixed)
#   make run        build and run everything at default sizes
#   make callgrind  instruction-level attribution
#
# For the before/after workflow use ./run_suite.sh, not these targets
# directly -- see README.md.

# clang++ 20 by default: bit-identical to g++ and 4.4% faster engine time at
# 20 threads (12.6% single-threaded), worklog entry 28. Needs libomp-20-dev.
# CXX=g++ on the command line or in the environment still selects GCC.
# (Make predefines CXX, so ?= would never apply; origin tells them apart.)
ifeq ($(origin CXX),default)
CXX := clang++-20
endif
SRC_DIR  := ../corintho_ai/cpp/src
INC      := -I../corintho_ai/cpp/include -I../gsl/include -I.
CXXFLAGS := -std=c++17 -O3 -DNDEBUG -g -flto -march=native -fopenmp -Wall -Wextra
# Header dependencies: every object also writes a .d file listing the headers
# it included, and those are pulled in below. Without this, changing a header
# left stale objects that LTO then reported as One Definition Rule violations.
DEPFLAGS := -MMD -MP
LDFLAGS  := -fopenmp -flto -O3 -march=native

BUILD := build
CORE  := $(BUILD)/util.o $(BUILD)/move.o $(BUILD)/game.o $(BUILD)/node.o \
         $(BUILD)/solver.o $(BUILD)/solver_pool.o \
         $(BUILD)/trainmc.o $(BUILD)/selfplayer.o $(BUILD)/trainer.o \
         $(BUILD)/mlp.o
# Built here only; not listed in corintho_ai/python/setup.py, so it never
# ships in the training module.
CORE  += $(BUILD)/game_reference.o

.PHONY: all micro selfplay selfplay_nn mlp_bench golden verify rulecheck parity run callgrind clean

all: micro selfplay selfplay_nn mlp_bench golden verify_bin rulecheck_bin parity

micro: $(BUILD)/micro_bench
selfplay: $(BUILD)/selfplay_bench
selfplay_nn: $(BUILD)/selfplay_nn
mlp_bench: $(BUILD)/mlp_bench
golden: $(BUILD)/golden
verify_bin: $(BUILD)/verify
rulecheck_bin: $(BUILD)/rulecheck
parity: $(BUILD)/parity

$(BUILD):
	@mkdir -p $(BUILD)

$(BUILD)/%.o: $(SRC_DIR)/%.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INC) -c $< -o $@

$(BUILD)/micro.o: micro.cpp bench_util.h | $(BUILD)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INC) -c $< -o $@

$(BUILD)/selfplay.o: selfplay.cpp bench_util.h | $(BUILD)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INC) -c $< -o $@

$(BUILD)/selfplay_nn.o: selfplay_nn.cpp bench_util.h | $(BUILD)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INC) -DBENCH_DIR='"$(CURDIR)"' -c $< -o $@

$(BUILD)/mlp_bench.o: mlp_bench.cpp bench_util.h | $(BUILD)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INC) -c $< -o $@

$(BUILD)/golden.o: golden.cpp bench_util.h | $(BUILD)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INC) -c $< -o $@

$(BUILD)/verify.o: verify.cpp bench_util.h | $(BUILD)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INC) -c $< -o $@

$(BUILD)/rulecheck.o: rulecheck.cpp bench_util.h | $(BUILD)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INC) -c $< -o $@

$(BUILD)/parity.o: parity.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INC) -c $< -o $@

$(BUILD)/micro_bench: $(BUILD)/micro.o $(CORE)
	$(CXX) $^ $(LDFLAGS) -o $@

$(BUILD)/selfplay_bench: $(BUILD)/selfplay.o $(CORE)
	$(CXX) $^ $(LDFLAGS) -o $@

$(BUILD)/selfplay_nn: $(BUILD)/selfplay_nn.o $(CORE)
	$(CXX) $^ $(LDFLAGS) -o $@

$(BUILD)/mlp_bench: $(BUILD)/mlp_bench.o $(CORE)
	$(CXX) $^ $(LDFLAGS) -o $@

$(BUILD)/golden: $(BUILD)/golden.o $(CORE)
	$(CXX) $^ $(LDFLAGS) -o $@

$(BUILD)/verify: $(BUILD)/verify.o $(CORE)
	$(CXX) $^ $(LDFLAGS) -o $@

$(BUILD)/rulecheck: $(BUILD)/rulecheck.o $(CORE)
	$(CXX) $^ $(LDFLAGS) -o $@

$(BUILD)/parity: $(BUILD)/parity.o $(CORE)
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

# Gate for every optimization: must pass before a stage is committed.
verify: $(BUILD)/verify
	@./$(BUILD)/verify

# Exhaustive over the proven-complete index. Expected to FAIL until the
# line-breaking defects in worklog/RULES-CHECKLIST.md item 4 are fixed.
rulecheck: $(BUILD)/rulecheck
	@./$(BUILD)/rulecheck

-include $(wildcard $(BUILD)/*.d)

clean:
	rm -rf $(BUILD) build-lines build-a build-b perf.data cg.out
