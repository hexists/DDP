CNN
===========
# Reference
 - http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
 - https://github.com/dennybritz/cnn-text-classification-tf
 - ref test: 
```
#set venv
python3 -m venv venv_py3_tf1.12_cpu
source venv_py3_tf1.12_cpu/bin/activate

# install tf "Current release for CPU-only"
pip install tensorflow
pip install --upgrade tensorflow==1.12
python -c 'import tensorflow as tf; print(tf.__version__)'

# install numpy
pip install numpy

# git checkout cnn-text-classification
git clone https://github.com/dennybritz/cnn-text-classification-tf.git
cd cnn-text-classification-tf

#Train
./train.py

#evaluation
./eval.py --eval_train --checkpoint_dir="./runs/1459637919/checkpoints/"

 - 참고) eval.py 오류: FLAGS._parse_flags()
 - 해결 방법: https://github.com/dennybritz/cnn-text-classification-tf/pull/165
 
```
