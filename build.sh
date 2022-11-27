# Format C++
clang-format -i --verbose cpp/*
# Format Python
black *.py
# Build Cython module
python3 setup.py build_ext --inplace