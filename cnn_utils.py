import tensorflow as tf
from tensorflow import data
from datetime import datetime
import multiprocessing
import shutil
import numpy as np

MULTI_THREADING = False
DATA_PATH = r'd:\WMIND\temp\train_data_cnn_small.csv'
HEADER = ['uid','ts','v','slot','label','seq']
HEADER_DEFAULTS = [[0], ['NA'], [0], [0], [0], ['NA']]
TARGET_NAME = 'label'

def parse_row(row):
  columns = tf.decode_csv(row, record_defaults=HEADER_DEFAULTS, field_delim=',')
  features = dict(zip(HEADER, columns))

  target = features.pop(TARGET_NAME)

  return features, target

def input_fn(files_name_pattern, mode=tf.estimator.ModeKeys.EVAL,
             skip_header_lines=0,
             num_epochs=1,
             batch_size=200):
  shuffle = True if mode == tf.estimator.ModeKeys.TRAIN else False

  num_threads = multiprocessing.cpu_count() if MULTI_THREADING else 1

  buffer_size = 2 * batch_size + 1

  print("")
  print("* data input_fn:")
  print("================")
  print("Input file(s): {}".format(files_name_pattern))
  print("Batch size: {}".format(batch_size))
  print("Epoch Count: {}".format(num_epochs))
  print("Mode: {}".format(mode))
  print("Thread Count: {}".format(num_threads))
  print("Shuffle: {}".format(shuffle))
  print("================")
  print("")

  file_names = tf.matching_files(files_name_pattern)
  dataset = data.TextLineDataset(filenames=file_names)

  dataset = dataset.skip(skip_header_lines)

  if shuffle:
    dataset = dataset.shuffle(buffer_size)

  dataset = dataset.map(lambda row: parse_row(row),
                        num_parallel_calls=num_threads)

  dataset = dataset.batch(batch_size)
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.prefetch(buffer_size)

  iterator = dataset.make_one_shot_iterator()

  features, target = iterator.get_next()
  return features, target


if __name__ == "__main__":
  features_op, labels_op = input_fn(DATA_PATH, mode=tf.estimator.ModeKeys.EVAL, batch_size=8)
  keys = tf.constant([1,3])
  values = tf.constant([[1.0, 2.0], [3.0, 4.0]])

  table = tf.contrib.lookup.HashTable(
    tf.contrib.lookup.KeyValueTensorInitializer(keys, values), -1
  )

  ii = tf.constant(1)
  out = table.lookup(ii)
  with tf.Session() as sess:
    # initialise and start the queues.
    sess.run(tf.local_variables_initializer())
    tf.tables_initializer().run()

    coordinator = tf.train.Coordinator()
    _ = tf.train.start_queue_runners(coord=coordinator)

    a1, a2 = sess.run([features_op, labels_op])
    a3 = sess.run(out)

    print(a1, a2)
    print(a3)
