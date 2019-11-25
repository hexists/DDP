# How to convert pretrained tf1 pretrained BERT to tf2.


*아래 방법대로 변환은 문제 없으나, 실행이 안되서 문제 확인중이다.*


tensorflow v1으로 저장된 BERT Model을 v2로 변환하는 방법에 대해 정리한다.   
(경험치를 쌓아가는 과정에 정리한 문서이다.)    

여기서 다루는 것은 check point에 대한 변환 방법이다.


[tensorflow/models/official/nlp/bert](https://github.com/tensorflow/models/tree/master/official/nlp/bert)를 보면 **tf1\_to_keras\_checkpoint\_converter.py** 이런 파일이 있는데, 이 파일을 통해 v1로 학습된 모델을 v2로 변환할 수 있다.

tf2\_checkpoint\_converter.py 파일은 v2에서 생성된 모델을 v1으로 변환할 때, 쓸 수 있다.    

실행 방법이다.

```
$ python tf1_to_keras_checkpoint_converter.py \
--checkpoint_from_path=/path/to/tf1_model/bert_model.ckpt \
--checkpoint_to_path=/path/to/tf2_model/bert_model.ckpt
```

실행 했을 때, 아래와 같은 내용이 출력된다.

```
...

Instructions for updating:
If using Keras pass *_constraint arguments to layers.

...

If using Keras pass *_constraint arguments to layers.
INFO:tensorflow:Converted: bert/encoder/layer_9/output/dense/kernel --> bert_model/encoder/layer_9/output/dense/kernel
I1125 08:32:48.756862 139821589006144 tf1_checkpoint_converter_lib.py:73] Converted: bert/encoder/layer_9/output/dense/kernel --> bert_model/encoder/layer_9/output/dense/kernel
INFO:tensorflow:Converted: bert_model/encoder/layer_9/output/dense/kernel --> bert_model/encoder/layer_9/output/kernel
I1125 08:32:48.757090 139821589006144 tf1_checkpoint_converter_lib.py:73] Converted: bert_model/encoder/layer_9/output/dense/kernel --> bert_model/encoder/layer_9/output/kernel
INFO:tensorflow:Converted: bert/encoder/layer_9/output/dense/bias --> bert_model/encoder/layer_9/output/dense/bias
I1125 08:32:48.833088 139821589006144 tf1_checkpoint_converter_lib.py:73] Converted: bert/encoder/layer_9/output/dense/bias --> bert_model/encoder/layer_9/output/dense/bias

...

I1125 08:33:13.800855 139821589006144 tf1_checkpoint_converter_lib.py:193] Summary:
INFO:tensorflow:  Converted 206 variable name(s).
I1125 08:33:13.801164 139821589006144 tf1_checkpoint_converter_lib.py:194]   Converted 206 variable name(s).
INFO:tensorflow:  Converted: {'bert/encoder/layer_9/output/dense/kernel': 'bert_model/encoder/layer_9/output/kernel', 'bert/encoder/layer_9/output/dense/bias': 'bert_model/encoder/layer_9/output/bias', 'bert/encoder/layer_9/output/LayerNorm/gamma': 'bert_model/encoder/layer_9/output_layer_norm/gamma', 'bert/encoder/layer_9/intermediate/dense/kernel': 'bert_model/encoder/layer_9/intermediate/kernel', 'bert/encoder/layer_9/intermediate/dense/bias': 'bert_model/encoder/layer_9/intermediate/bias', 'bert/encoder/layer_9/attention/self/query/bias': 'bert_model/encoder/layer_9/self_attention/query/bias', 'bert/encoder/layer_9/attention/self/key/bias': 'bert_model/encoder/layer_9/self_attention/key/bias', 'bert/encoder/layer_9/attention/output/LayerNorm/beta': 'bert_model/encoder/layer_9/self_attention_layer_norm/beta', 'bert/encoder/layer_8/output/LayerNorm/beta': 'bert_model/encoder/layer_8/output_layer_norm/beta', 'bert/

...
```
