dos2unix *
# Build Cython module
cd corintho_ai/python
python3 setup_tourney.py build_ext --inplace -j 4
python3 tourney -p ../toml/tourney.toml -n ${1:-1}