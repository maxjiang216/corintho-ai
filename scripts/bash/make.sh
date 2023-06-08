dos2unix cpp/*
# Format C++
clang-format -i --verbose cpp/*.cpp cpp/*.h
cd cpp
make
make clean
cd ..