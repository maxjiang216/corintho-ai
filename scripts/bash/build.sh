dos2unix *
# Build Cython module
cd corintho_ai/python
python3 setup.py build_ext --inplace -j 4
rm main.cpp
cd ../..