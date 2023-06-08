dos2unix *
# Format C++
clang-format -i --verbose cpp/*.cpp cpp/*.h
cd python
# Format Python
black *.py
# Build Cython module
python3 setup.py build_ext --inplace -j 4
rm main.cpp
cd ..