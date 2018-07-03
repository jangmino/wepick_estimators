from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import data
from datetime import datetime
import multiprocessing
import shutil
import numpy as np
import csv
import pickle
import os

import cnn_utils

tf.enable_eager_execution()

# 강제로 GPU 사용안하게 함
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def get_raw_data(input_csv_file, deal_to_w2vec_file=r'd:\WMIND\temp\deal_to_w2vec.pkl'):
  """
  검증용.
  :param input_csv_file:
  :param deal_to_w2vec_file:
  :return:
  """
  with open(deal_to_w2vec_file, 'rb') as f:
    deal_word2vec_dic = pickle.load(f)

  deal_word2vec_bytes = {}
  for did, vec in deal_word2vec_dic.items():
    deal_word2vec_bytes[did] = vec.tobytes()

  raw_datas = []
  for i, row in enumerate(cnn_utils.creat_csv_iterator(input_csv_file)):
    raw_data = cnn_utils.create_example(deal_word2vec_bytes, row,verify=True)
    raw_datas.append(raw_data)

  return raw_datas


class TFRecordTest(tf.test.TestCase):

  def testPaddedBatch(self):
    # padded_batch 결과가 의도한 대로 만들어졌는지 테스트
    raw_datas = get_raw_data(r'd:\WMIND\temp\train_data_cnn_small.csv')
    features_op, labels_op = cnn_utils.tfrecords_input_fn(r'd:\WMIND\temp\train_data_cnn_small.tfrecords',
                                                mode=tf.estimator.ModeKeys.EVAL, batch_size=4)

    for i in range(4):
      x = [np.frombuffer(x, dtype=np.float32) for x in raw_datas[i]['org'][2]]
      y = np.vstack(x)
      z = features_op['w2vecs'][i]
      self.assertAllClose(y, z[0:y.shape[0]])
      self.assertAllClose(z[y.shape[0]:], np.zeros(z[y.shape[0]:].shape))

    pass

if __name__ == "__main__":
  tf.test.main()
