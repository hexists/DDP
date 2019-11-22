# BERT in TF2.0

- BERT를 tf2.0에서 사용하는 방법을 정리합니다.
- [tensorflow/models](https://github.com/tensorflow/models/tree/master/official/nlp/bert)에 있는 내용을 바탕으로 정리합니다.


## Repo Url에 따른 차이

- BERT에 대해 찾아보면 여러 repo가 보입니다.
- [https://github.com/google-research/bert](https://github.com/google-research/bert)
- [https://github.com/tensorflow/models/tree/master/official/nlp/bert](https://github.com/tensorflow/models/tree/master/official/nlp/bert)
- google-research: BERT 논문에 대한 코드와 모델입니다.
- official/models: tf 버전에 맞춰 개발되는 곳입니다.

## 전체 Flow
run_classifier.py 실행을 위한 내용입니다.

1. models path를 설정합니다.
2. tf-nightly를 설치합니다.
3. BERT model을 다운로드 합니다.
4. GLUE data를 다운로드 합니다.
5. tf_record로 데이터를 변환합니다.
6. fine tunning을 수행합니다(run_classifer.py).


모델을 실행하려면 PYTHONPATH에 models path를 추가해야 합니다.

```
export PYTHONPATH="$PYTHONPATH:/path/to/models"
```

그리고 tf-nightly-gpu 를 설치합니다.   
(tf-nightly는 develop인 버전을 의미합니다. 왜 이걸 설치해야 하는지는 잘 모르겠네요.)

```
$ pip install tf-nightly-gpu
```

BERT model을 다운로드 합니다.    
TF2용으로 빌드된 모델을 받아야 합니다.    
[BERT-Large, Uncased (Whole Word Masking)](https://storage.googleapis.com/cloud-tpu-checkpoints/bert/tf_20/wwm_uncased_L-24_H-1024_A-16.tar.gz) 저는 이 모델을 사용했습니다.

```
wget https://storage.googleapis.com/cloud-tpu-checkpoints/bert/tf_20/wwm_uncased_L-24_H-1024_A-16.tar.gz

tar xvzf wwm_uncased_L-24_H-1024_A-16.tar.gz
```

GLUE data를 다운로드 합니다. [url](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e)

```
$ python download_glue_data.py --data_dir glue_data --tasks all
```

fine tunning을 위한 데이터를 생성합니다.
몇몇 변수를 설정하고, 실행하면 됩니다.

```
export GLUE_DIR=/path/to/glue
export BERT_BASE_DIR=/path/to/bert

export TASK_NAME=MRPC  # task
export OUTPUT_DIR=/path/to/tf_record

python create_finetuning_data.py \
 --input_data_dir=${GLUE_DIR}/${TASK_NAME}/ \
 --vocab_file=${BERT_BASE_DIR}/vocab.txt \
 --train_data_output_path=${OUTPUT_DIR}/${TASK_NAME}_train.tf_record \
 --eval_data_output_path=${OUTPUT_DIR}/${TASK_NAME}_eval.tf_record \
 --meta_data_file_path=${OUTPUT_DIR}/${TASK_NAME}_meta_data \
 --fine_tuning_task_type=classification --max_seq_length=128 \
 --classification_task_name=${TASK_NAME}
```

fine tunning을 실행합니다.

```
export BERT_BASE_DIR=/path/to/bert
export MODEL_DIR=/path/to/output_model
export GLUE_DIR=/path/to/tf_record
export TASK=MRPC  # task

python run_classifier.py \
  --mode='train_and_eval' \
  --input_meta_data_path=${GLUE_DIR}/${TASK}_meta_data \
  --train_data_path=${GLUE_DIR}/${TASK}_train.tf_record \
  --eval_data_path=${GLUE_DIR}/${TASK}_eval.tf_record \
  --bert_config_file=${BERT_BASE_DIR}/bert_config.json \
  --init_checkpoint=${BERT_BASE_DIR}/bert_model.ckpt \
  --train_batch_size=4 \
  --eval_batch_size=4 \
  --steps_per_loop=1 \
  --learning_rate=2e-5 \
  --num_train_epochs=3 \
  --model_dir=${MODEL_DIR}

```


## 참고

- https://github.com/tensorflow/models/commits/master/official/nlp/bert/README.md, 252e63849b5cf8dd12d6930dc2e9f8c51ea70251
