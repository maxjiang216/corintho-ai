black ../../corintho_ai/python/*.py ../../corintho_ai/docker/*.py --line-length 79
isort ../../corintho_ai/python/*.py ../../corintho_ai/python/*.pyx ../../corintho_ai/docker/*.py ../../corintho_ai/docker/*.pyx
clang-format -i --verbose ../../corintho_ai/cpp/src/*.cpp ../../corintho_ai/cpp/include/*.h ../../tests/cpp/*.cpp