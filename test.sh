bash build.sh > build_out.txt
python3 main.py wrapper.py -p test.toml -n ${1:-1} > loss.txt