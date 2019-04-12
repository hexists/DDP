import tensorflow as tf
import numpy as np
import os
import time
import datetime
from train import preprocess
from cnn import CNN
import csv
## import function
from data import load_xy, batch_iter
from common import get_tfconfig
from params import BATCH_SIZE, EVAL_TRAIN

dir_name = "1551185115"
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", dir_name))
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
vocab_path = os.path.join(out_dir, "./", "vocab")

# CHANGE THIS: Load data. Load your own data here
if EVAL_TRAIN:
    x_raw, y_test, max_length = load_xy()
    y_test = np.argmax(y_test, axis=1)
else:
    x_raw = ["a masterpiece four years in the making", "everything is off."]
    y_test = [1, 0]

# Map data into vocabulary
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nEvaluating...\n")

# Evaluation
# ==================================================
graph = tf.Graph()
with graph.as_default():
    config = get_tfconfig()
    sess = tf.Session(config=config)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        x = graph.get_operation_by_name("x").outputs[0]
        keep_prob = graph.get_operation_by_name("keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predict").outputs[0]

        # Generate batches for one epoch
        batches = batch_iter(list(x_test), BATCH_SIZE, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {x: x_test_batch, keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
out_path = os.path.join(checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)
