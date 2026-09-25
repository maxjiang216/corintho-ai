# GPU pipeline driver (worklog entry 29). Run ./setup.sh first: it creates
# .venv (PyTorch, ONNX Runtime, and the CUDA/cuDNN libraries both use) and
# unpacks the ONNX Runtime C++ release into third_party/.
#
#   make            build build/corintho_play
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
RPATHS   := $(abspath $(ORT)/lib) $(NVIDIA)/cu13/lib $(NVIDIA)/cudnn/lib
LDFLAGS  := -fopenmp -flto -O3 -march=native -L$(ORT)/lib -lonnxruntime \
            -Wl,--disable-new-dtags $(foreach p,$(RPATHS),-Wl,-rpath,$(p))

BUILD := build
CORE  := util move game node trainmc selfplayer trainer mlp
OBJS  := $(CORE:%=$(BUILD)/%.o) $(BUILD)/backend.o $(BUILD)/corintho_play.o

.PHONY: all clean
all: $(BUILD)/corintho_play

$(BUILD):
	@mkdir -p $(BUILD)

$(BUILD)/%.o: $(SRC_DIR)/%.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INC) -c $< -o $@

$(BUILD)/%.o: cpp/%.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INC) -c $< -o $@

$(BUILD)/corintho_play: $(OBJS)
	$(CXX) $^ $(LDFLAGS) -o $@

-include $(wildcard $(BUILD)/*.d)

clean:
	rm -rf $(BUILD)
