sudo apt-get update
sudo apt install python3-pip -y # 328MB
sudo apt-get install git -y # 39MB
pip install numpy # 17MB
pip install keras==2.9 # 5MB
pip install tf-nightly # 588MB
pip install cython # 5MB
export PATH="~/.local/bin:$PATH"
git clone https://github.com/maxjiang216/corintho-ai.git
cd corintho-ai