bash scripts/bash/build.sh
python3 corintho_ai/python/main.py corintho_ai/python/wrapper.py -p corintho_ai/toml/train.toml -n ${1:-1}
