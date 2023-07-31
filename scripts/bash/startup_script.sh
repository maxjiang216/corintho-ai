sudo apt-get update
sudo apt install python3-pip -y # 328MB
sudo apt-get install git -y # 39MB
sudo apt-get install clang-format -y
sudo apt-get install dos2unix -y
pip install numpy # 17MB
pip install keras==2.9 # 5MB
pip install tf-nightly # 588MB
pip install pytz
pip install cython # 5MB
pip install toml
pip install black
pip install tflite_runtime
export PATH="~/.local/bin:$PATH"
git clone https://github.com/maxjiang216/corintho-ai.git
cd corintho-ai
git clone https://github.com/microsoft/GSL.git gsl
dos2unix *
chmod +x scripts/bash/train.sh
nohup ~/corintho-ai/scripts/bash/train.sh &
