cd corintho_ai/docker
python3 setup.py build_ext --inplace -j4
cd ../..
docker build . -f corintho_ai/docker/Dockerfile -t corintho-play
docker run -p 8080:8080 corintho-play