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
