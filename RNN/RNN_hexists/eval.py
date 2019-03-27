#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_rnn import TextRNN
from tensorflow.contrib import learn
import csv
import argparse

# Parameters
# ==================================================

parser = argparse.ArgumentParser()

#Data Parameters
parser.add_argument('--positive_data_file', type=str, default='./data/rt-polaritydata/rt-polarity.pos', help='Data source for the positive data.')
parser.add_argument('--negative_data_file', type=str, default='./data/rt-polaritydata/rt-polarity.neg', help='Data source for the positive data.')

#Eval Parameters
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size (default: 64).')
parser.add_argument('--checkpoint_dir', type=str, default=None, help='Checkpoint directory from training run.')
parser.add_argument('--eval_train', type=bool, default=False, help='Evaluate on all training data.')

#Misc Parameters
parser.add_argument('--allow_soft_placement', type=bool, default=True, help='Allow device soft device placement.')
parser.add_argument('--log_device_placement', type=bool, default=False, help='Log placement of ops on devices.')

FLAGS = parser.parse_args()

print("\nParameters:")
for attr in vars(FLAGS):
    print("{}={}".format(attr.upper(), getattr(FLAGS, attr)))
print("")

# CHANGE THIS: Load data. Load your own data here
if True:
    x_raw, y_test = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
    y_test = np.argmax(y_test, axis=1)
else:
    x_raw = ["a masterpiece four years in the making", "everything is off."]
    y_test = [1, 0]

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions.extend(batch_predictions)

# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(x_raw), y_test, all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f, delimiter='\t').writerows(predictions_human_readable)
