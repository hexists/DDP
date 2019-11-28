# BERT in TF1.x

- BERT를 tf1.11.0에서 실행한 내용입니다.
- tf2 실행은 [여기](https://github.com/hexists/DDP/blob/master/BERT/simple_bert_using_tf2/README.fine_tunning.md)를 참고하세요.

## Reference

- [https://github.com/google-research/bert#prediction-from-classifier](https://github.com/google-research/bert#prediction-from-classifier)
- 여기 있는 내용을 그대로 실행했습니다.

## Model

- [BERT-Base, Multilingual Cased](https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip)을 이용했습니다.
- 이 모델을 사용할 때는 **--do\_lower\_case=False**를 설정해야 합니다.

## Fine-tunning with BERT(task = MRPC)
- [Sentence (and sentence-pair) classification tasks
](https://github.com/google-research/bert#sentence-and-sentence-pair-classification-tasks)

- run_cls.sh

  ```
  #!/bin/bash

  CDIR=$(readlink -f $(dirname $(readlink -f ${BASH_SOURCE[0]})))

  TASK=${1:-"MRPC"}

  PYTHONPATH="$PYTHONPATH:$CDIR/bert"

  BERT_BASE_DIR=$CDIR/multi_cased_L-12_H-768_A-12
  GLUE_DIR=$CDIR/glue_data
  OUTPUT_DIR=$CDIR/output

  echo ""
  echo "TASK         : $TASK"
  echo "BERT_BASE_DIR: $BERT_BASE_DIR"
  echo "GLUE_DIR     : $GLUE_DIR"
  echo "OUTPUT_DIR   : $OUTPUT_DIR"

  echo "pushd bert
  time CUDA_VISIBLE_DEVICES=0 \
  python run_classifier.py \
  --task_name=$TASK \
  --do_train=true \
  --do_eval=true \
  --data_dir=$GLUE_DIR/$TASK \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$OUTPUT_DIR/mrpc_output/ \
  --do_lower_case=False
  popd bert"
  pushd bert
  time CUDA_VISIBLE_DEVICES=0 \
  python run_classifier.py \
  --task_name=$TASK \
  --do_train=true \
  --do_eval=true \
  --data_dir=$GLUE_DIR/$TASK \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$OUTPUT_DIR/mrpc_output/ \
  --do_lower_case=False
  popd bert
  ```


## Prediction from classifier
- [Prediction from classifier
](https://github.com/google-research/bert#prediction-from-classifier)

  ```
  #!/bin/bash

  CDIR=$(readlink -f $(dirname $(readlink -f ${BASH_SOURCE[0]})))

  TASK=${1:-"MRPC"}

  PYTHONPATH="$PYTHONPATH:$CDIR/bert"

  BERT_BASE_DIR=$CDIR/multi_cased_L-12_H-768_A-12
  GLUE_DIR=$CDIR/glue_data
  OUTPUT_DIR=$CDIR/output
  TRAINED_CLASSIFIER=$OUTPUT_DIR/mrpc_output
  PRED_DIR=$CDIR/pred

  echo ""
  echo "TASK               : $TASK"
  echo "BERT_BASE_DIR      : $BERT_BASE_DIR"
  echo "GLUE_DIR           : $GLUE_DIR"
  echo "TRAINED_CLASSIFIER : $TRAINED_CLASSIFIER"
  echo "PRED_DIR           : $PRED_DIR"

  echo ""
  echo "pushd bert
  time CUDA_VISIBLE_DEVICES=0 \
  python run_classifier.py \
  --task_name=$TASK \
  --do_predict=true \
  --data_dir=$GLUE_DIR/$TASK \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$TRAINED_CLASSIFIER \
  --max_seq_length=128 \
  --output_dir=$PRED_DIR/mrpc_output/ \
  --do_eval=true \
  --do_lower_case=False
  popd bert"
  pushd bert
  time CUDA_VISIBLE_DEVICES=0 \
  python run_classifier.py \
  --task_name=$TASK \
  --do_predict=true \
  --data_dir=$GLUE_DIR/$TASK \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$TRAINED_CLASSIFIER \
  --max_seq_length=128 \
  --output_dir=$PRED_DIR/mrpc_output/ \
  --do_eval=true \
  --do_lower_case=False
  popd bert
  ```

## 전체 과정

- 가상 환경 설정

```
$ virtualenv -p python3 .venv

$ pip install requirements.txt
```

- bert repo 클론

```
$ git clone https://github.com/google-research/bert.git
```


- bert model download

```
$ wget https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip

$ unzip multi_cased_L-12_H-768_A-12.zip
```

- glue data 다운로드

```
$ git clone https://gist.github.com/60c2bdb54d156a41194446737ce03e2e.git download_glue

$ python download_glue/download_glue_data.py
```

- fine tunning

```
$ sh run_cls.sh
```

- prediction

```
sh pred_cls.sh
```
