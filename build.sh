dos2unix *
# Format C++
clang-format -i --verbose cpp/*.cpp cpp/*.h
# Format Python
black *.py
# Build Cython module
#OPT="-DNDEBUG -Ofast -Wall" \
#CFLAGS="-Wno-unused-result -Wsign-compare -DNDEBUG -Wall -ffile-prefix-map=/build/python3.9-RNBry6/python3.9-3.9.2=. -fno-stack-protector -Wformat -Werror=format-security" \
#LDFLAGS="-Wl,-z,relro -Ofast" \
#LDSHARED="x86_64-linux-gnu-gcc -pthread -shared -Wl,-Ofast -Wl,-Bsymbolic-functions -Wl,-z,relro -Ofast" \
python3 setup.py build_ext --inplace -j 4
rm main.cpp 