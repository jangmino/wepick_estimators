import tensorflow as tf
from tensorflow import data
from datetime import datetime
import multiprocessing
import shutil
import numpy as np
import csv
import pickle

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
  context = tf.train.Features(feature={
    'uid': _int64_feature(int(data['uid'])),
    'ts': _bytes_feature(data['ts'].encode('utf-8')),
    'label': _int64_feature(int(data['label']))
    }
  )

  vecs = []
  dids = []
  for did_ in reversed(data['seq'].split(' ')):
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


def create_tfrecords_file(input_csv_file, deal_to_w2vec_file=r'd:\WMIND\temp\deal_to_w2vec.pkl'):
  with open(deal_to_w2vec_file, 'rb') as f:
    deal_word2vec_dic = pickle.load(f)

  deal_word2vec_bytes = {}
  for did, vec in deal_word2vec_dic.items():
    deal_word2vec_bytes[did] = vec.tobytes()

  output_tfrecord_file = input_csv_file.replace("csv", "tfrecords")
  writer = tf.python_io.TFRecordWriter(output_tfrecord_file)
  raw_datas = []
  for i, row in enumerate(creat_csv_iterator(input_csv_file)):
    example = create_example(deal_word2vec_bytes, row)
    content = example.SerializeToString()
    writer.write(content)
  writer.close()


def parse_tf_example(example_proto):
  """
  w2vecs를
  - tf.decode_raw를 적용하여, [H, D] 로 reshape
  :param example_proto:
  :return:
  """
  # Define features
  context_features = {
    'uid': tf.FixedLenFeature([], dtype=tf.int64),
    'ts': tf.FixedLenFeature([], dtype=tf.string),
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
  data['w2vecs'] = tf.reshape( tf.decode_raw(sequence_data['w2vecs'], tf.float32), [-1, DIM])
  data['dids'] = sequence_data.pop('dids')
  data['uid'] = context_data.pop('uid')
  # data['ts'] = context_data.pop('ts')
  target = context_data.pop('label')

  return data, target


def tfrecords_input_fn(files_name_pattern, mode=tf.estimator.ModeKeys.EVAL,
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
  print("================")
  print("")

  file_names = tf.matching_files(files_name_pattern)
  dataset = data.TFRecordDataset(filenames=file_names)

  if shuffle:
    dataset = dataset.shuffle(buffer_size)

  dataset = dataset.map(lambda tf_example: parse_tf_example(tf_example))
  dataset = dataset.padded_batch(batch_size, padded_shapes=({'uid':[], 'w2vecs':[None, DIM], 'dids':[None]}, []) )

  dataset = dataset.repeat(num_epochs)
  dataset = dataset.prefetch(buffer_size)

  iterator = dataset.make_one_shot_iterator()

  features, target = iterator.get_next()
  return features, target


if __name__ == "__main__":
  tf.enable_eager_execution()
  create_tfrecords_file(r'd:\WMIND\temp\train_data_cnn_small.csv')

  features_op, labels_op = tfrecords_input_fn(r'd:\WMIND\temp\train_data_cnn_small.tfrecords', mode=tf.estimator.ModeKeys.EVAL, batch_size=4)

  print(features_op, labels_op)