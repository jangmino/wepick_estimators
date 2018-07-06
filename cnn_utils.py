import tensorflow as tf
from tensorflow import data
from datetime import datetime
import multiprocessing
import shutil
import numpy as np
import csv
import pickle
import os

MULTI_THREADING = False
DATA_PATH = r'd:\WMIND\temp\train_data_cnn_small.csv'
HEADER = ['uid','ts','v','slot','label','seq']
HEADER_DEFAULTS = [[0], ['NA'], [0], [0], [0], ['NA']]
TARGET_NAME = 'label'
HISTORY_NAME = 'seq'
MAX_DOCUMENT_LENGTH = 16
DIM = 100


def _int64_feature(value):
  """Wrapper for inserting an int64 Feature into a SequenceExample proto,
e.g, An integer label.
"""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  """Wrapper for inserting a bytes Feature into a SequenceExample proto,
e.g, an image in byte
"""
  # return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature_list(values):
  """Wrapper for inserting an int64 FeatureList into a SequenceExample proto,
e.g, sentence in list of ints
"""
  return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _bytes_feature_list(values):
  """Wrapper for inserting a bytes FeatureList into a SequenceExample proto,
e.g, sentence in list of bytes
"""
  return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])


def creat_csv_iterator(csv_file_path):
  with open(csv_file_path) as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
      yield row


def create_example(deal_word2vec_bytes, row, verify=False):
  """
  dids와 wvecs를 구성
  - 편의상 reversed를 함 (거꾸로 만듬)
  - did -> np.float32 벡터를 tobytes()를 사용후 bytes_list 로 저장
  - SequenceExample로 저장
  :param deal_word2vec_bytes:
  :param row:
  :param verify:
  :return:
  """
  data = dict(zip(HEADER, row))

  reversed_dids = list(reversed(data['seq'].split(' ')))
  context = tf.train.Features(feature={
    'uid': _int64_feature(int(data['uid'])),
    #'ts': _bytes_feature(data['ts'].encode('utf-8')),
    'sl': _int64_feature(len(reversed_dids)),
    'label': _int64_feature(int(data['label']))
    }
  )

  vecs = []
  dids = []
  for did_ in reversed_dids:
    did = int(did_)
    if did in deal_word2vec_bytes:
      vec = deal_word2vec_bytes[did]
      vecs.append(vec)
      dids.append(did)

  feature_lists = tf.train.FeatureLists(feature_list={
    'dids': _int64_feature_list(dids),
    'w2vecs': _bytes_feature_list(vecs)
    }
  )

  return {'org': (data['uid'], dids, vecs)} if verify else tf.train.SequenceExample(context=context,
                                                                                    feature_lists=feature_lists)


def create_tfrecords_file(input_csv_file, deal_to_w2vec_file=r'd:\WMIND\temp\deal_to_w2vec.pkl', out_header='data', num_in_one_file=320000):
  with open(deal_to_w2vec_file, 'rb') as f:
    deal_word2vec_dic = pickle.load(f)

  deal_word2vec_bytes = {}
  for did, vec in deal_word2vec_dic.items():
    deal_word2vec_bytes[did] = vec.tobytes()

  writer = None
  num_data_files = 0
  for i, row in enumerate(creat_csv_iterator(input_csv_file)):
    if writer is None or i % num_in_one_file == 0:
      if writer: writer.close()
      num_data_files += 1
      data_path = os.path.join(out_header, 'data-{:04d}.tfrecords'.format(num_data_files))
      writer = tf.python_io.TFRecordWriter(data_path)
      print("...creating {}.".format(data_path))
    example = create_example(deal_word2vec_bytes, row)
    content = example.SerializeToString()
    writer.write(content)
  if writer: writer.close()


def parse_tf_example(example_proto, max_history_len=None):
  """
  w2vecs를
  - tf.decode_raw를 적용하여, [H, D] 로 reshape
  :param example_proto:
  :return:
  """
  # Define features
  context_features = {
    'uid': tf.FixedLenFeature([], dtype=tf.int64),
#    'ts': tf.FixedLenFeature([], dtype=tf.string),
    'sl': tf.FixedLenFeature([], dtype=tf.int64),
    'label': tf.FixedLenFeature([], dtype=tf.int64)
  }
  sequence_features = {
    'dids': tf.FixedLenSequenceFeature([], dtype=tf.int64),
    'w2vecs': tf.FixedLenSequenceFeature([], dtype=tf.string),
  }

  # Extract features from serialized data
  context_data, sequence_data = tf.parse_single_sequence_example(
    serialized=example_proto,
    context_features=context_features,
    sequence_features=sequence_features)

  data = {}
#  data['w2vecs'] = tf.reshape( tf.decode_raw(sequence_data['w2vecs'], tf.float32), [-1, DIM])
  x_ = tf.reshape(tf.decode_raw(sequence_data['w2vecs'], tf.float32), [-1, DIM])
  if max_history_len is not None:
    padding = tf.constant([[0,max_history_len], [0, 0]])
    x_ = tf.pad(x_, padding)
    x_ = tf.slice(x_, [0,0], [max_history_len, -1])
  data['w2vecs'] = x_

  data['dids'] = sequence_data.pop('dids')
  data['uid'] = context_data.pop('uid')
  data['sl'] = context_data.pop('sl')
  # data['ts'] = context_data.pop('ts')
  target = context_data.pop('label')

  return data, target


def tfrecords_input_fn(files_name_pattern, mode=tf.estimator.ModeKeys.EVAL,
             max_history_len=None,
             num_epochs=1,
             batch_size=200):
  """
  padded_batch 를 적용
  - dict 내의 각 피쳐에 대해 Shape를 지정
  - padding은 기본값인 0.0으로 채워짐

  :param files_name_pattern:
  :param mode:
  :param num_epochs:
  :param batch_size:
  :return:
  """
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
  if max_history_len is not None:
    print("max_history_len: {}".format(max_history_len))
  print("================")
  print("")

  file_names = tf.matching_files(files_name_pattern)
  dataset = data.TFRecordDataset(filenames=file_names)

  if shuffle:
    dataset = dataset.shuffle(buffer_size)

  dataset = dataset.map(lambda tf_example: parse_tf_example(tf_example,max_history_len))
  dataset = dataset.padded_batch(batch_size, padded_shapes=({'uid':[], 'w2vecs':[None, DIM], 'dids':[None], 'sl':[]}, []) )

  dataset = dataset.repeat(num_epochs)
  dataset = dataset.prefetch(buffer_size)

  iterator = dataset.make_one_shot_iterator()

  features, target = iterator.get_next()
  return features, target


if __name__ == "__main__":
  tf.enable_eager_execution()
  # create_tfrecords_file(r'd:\WMIND\temp\train_data_cnn.csv', out_header=r'd:\WMIND\temp\train_data')
  # create_tfrecords_file(r'd:\WMIND\temp\test_data_cnn.csv', out_header=r'd:\WMIND\temp\test_data')

  features_op, labels_op = tfrecords_input_fn(r'd:\WMIND\temp\train_data_cnn_small.tfrecords', mode=tf.estimator.ModeKeys.TRAIN, batch_size=1)
  # features_op, labels_op = tfrecords_input_fn(r'c:\Users\wmp\TensorFlow\wepick_estimators\data\data-*.tfrecords',
  #                                             mode=tf.estimator.ModeKeys.EVAL, batch_size=8)
  #
  print(features_op, labels_op)