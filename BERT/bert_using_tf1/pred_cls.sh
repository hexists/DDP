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
