CNN
===========
# Reference
 - https://towardsdatascience.com/lstm-by-example-using-tensorflow-feb0c1968537
 - https://raw.githubusercontent.com/roatienza/Deep-Learning-Experiments/master/Experiments/Tensorflow/RNN/rnn_words.py
 - https://www.guru99.com/rnn-tutorial.html
 
# Requirements
 - Python 3
 - Tensorflow > 1.12
 - Numpy
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
```
 
# training
Train:
```
./train.py
```
tensorboard:
```
tensorboard --logdir ./runs/1550584483/summaries --port 7788
```
# Evaluating
```
eval.py --checkpoint_dir ./runs/1550584483/checkpoints

Total number of test examples: 10662
Accuracy: xxxx
Saving evaluation to ./runs/1550584483/checkpoints/../prediction.csv
```
```
cat ./runs/1550584483/checkpoints/../prediction.csv

#sentence    y    y'
middle earth    1       1
effective but too tepid biopic  1       1
```
