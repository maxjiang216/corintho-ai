# GPU pipeline driver (worklog entry 29). Run ./setup.sh first: it creates
# .venv (PyTorch, ONNX Runtime, and the CUDA/cuDNN libraries both use) and
# unpacks the ONNX Runtime C++ release into third_party/.
#
#   make            build build/corintho_play
#   make build/ab_match   alpha-beta vs MCTS match tool (entry 22)
#   make build/ab_selfplay   alpha-beta self-play positions (entry 25)
#   make clean

ifeq ($(origin CXX),default)
CXX := clang++-20
endif

ORT      := third_party/onnxruntime-linux-x64-gpu_cuda13-1.30.0
NVIDIA   := $(abspath .venv/lib/python3.12/site-packages/nvidia)
SRC_DIR  := ../corintho_ai/cpp/src
INC      := -I../corintho_ai/cpp/include -I../gsl/include -I$(ORT)/include -Icpp
# Same release flags as bench/Makefile
CXXFLAGS := -std=c++17 -O3 -DNDEBUG -g -flto -march=native -fopenmp -Wall -Wextra
DEPFLAGS := -MMD -MP
# The CUDA provider dlopens cuDNN, cuBLAS and the CUDA runtime at run time.
# They come from the pip wheels in .venv, so the executable carries an
# old-style DT_RPATH (not RUNPATH), which also applies to libraries' dlopen.
# TensorRT (trt: and trt16: models) is the tensorrt-cu13 wheel's libraries.
RPATHS   := $(abspath $(ORT)/lib) $(NVIDIA)/cu13/lib $(NVIDIA)/cudnn/lib \
            $(abspath .venv/lib/python3.12/site-packages/tensorrt_libs)
LDFLAGS  := -fopenmp -flto -O3 -march=native -L$(ORT)/lib -lonnxruntime \
            -Wl,--disable-new-dtags $(foreach p,$(RPATHS),-Wl,-rpath,$(p))

BUILD := build
CORE  := util move game node trainmc selfplayer trainer mlp solver solver_pool
OBJS  := $(CORE:%=$(BUILD)/%.o) $(BUILD)/backend.o $(BUILD)/corintho_play.o

.PHONY: all clean
all: $(BUILD)/corintho_play $(BUILD)/libcorintho_features.so

$(BUILD):
	@mkdir -p $(BUILD)

$(BUILD)/%.o: $(SRC_DIR)/%.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INC) -c $< -o $@

$(BUILD)/%.o: cpp/%.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INC) -c $< -o $@

$(BUILD)/corintho_play: $(OBJS)
	$(CXX) $^ $(LDFLAGS) -o $@

# Alpha-beta with a small value network against MCTS, on the CPU (worklog
# entry 22); not part of `all`
$(BUILD)/ab_match: $(BUILD)/ab_match.o $(CORE:%=$(BUILD)/%.o)
	$(CXX) $^ $(LDFLAGS) -o $@

# Alpha-beta self-play (worklog entry 25): CPU only, no ONNX Runtime; not
# part of `all`
$(BUILD)/ab_selfplay: $(BUILD)/ab_selfplay.o $(CORE:%=$(BUILD)/%.o)
	$(CXX) $^ -fopenmp -flto -O3 -march=native -o $@

# Rules-derived features for pipeline/arch (legal-move masks, line spaces),
# loaded by Python through ctypes; position-independent, so built separately
FEATURE_SRC := $(SRC_DIR)/game.cpp $(SRC_DIR)/move.cpp $(SRC_DIR)/util.cpp \
               cpp/features.cpp
$(BUILD)/libcorintho_features.so: $(FEATURE_SRC) | $(BUILD)
	$(CXX) -std=c++17 -O2 -DNDEBUG -fPIC -shared -fopenmp $(INC) $^ -o $@

-include $(wildcard $(BUILD)/*.d)

clean:
	rm -rf $(BUILD)
