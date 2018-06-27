import argparse
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.python import debug as tf_debug
import sys
import os
import pathlib

#tf.logging.set_verbosity(tf.logging.INFO)

def load_train_data(D=100):
  base_header = ["u", "ts", "v", "slot", "label"]
  dtypes = {"u": np.int32, "ts": str, "v": np.int32, "slot": np.int32, "label": np.int32}
  name_header = base_header
  for i in range(D):
    field = "h{}".format(i)
    name_header.append(field)
    dtypes[field] = np.float32
  for i in range(D):
    field = "d{}".format(i)
    name_header.append(field)
    dtypes[field] = np.float32

  train_df = pd.read_csv(FLAGS.train, header=None, names=name_header, dtype=dtypes,
                         parse_dates=['ts'])

  col_dic = dict(zip(list(train_df), range(len(list(train_df)))))

  x_columns = list(train_df.columns[col_dic['h0']:col_dic['d99'] + 1])
  x = train_df[x_columns]
  y = train_df['label']

  feature_cols = [tf.feature_column.numeric_column(k) for k in x_columns]

  return x.values, y.values, feature_cols

def load_eval_data(D=100):
  test_base_header = ["u", "ts", "p_v", "p_slot", "n_v", "n_slot"]
  test_dtypes = {"u": np.int32, "ts": str, "p_v": np.int32, "p_slot": np.int32, "n_v": np.int32, "n_slot": np.int32}
  name_header = test_base_header
  for i in range(D):
    field = "h{}".format(i)
    name_header.append(field)
    test_dtypes[field] = np.float32
  for i in range(D):
    field = "p{}".format(i)
    name_header.append(field)
    test_dtypes[field] = np.float32
  for i in range(D):
    field = "n{}".format(i)
    name_header.append(field)
    test_dtypes[field] = np.float32

  test_df = pd.read_csv(FLAGS.eval, header=None, names=name_header, dtype=test_dtypes,
                        parse_dates=['ts'])

  col_dic = dict(zip(list(test_df), range(len(list(test_df)))))

  pos_columns = list(test_df.columns[col_dic['h0']:col_dic['p99'] + 1])
  neg_columns = list(test_df.columns[col_dic['h0']:col_dic['h99'] + 1]) + list(
    test_df.columns[col_dic['n0']:col_dic['n99'] + 1])

  x1 = test_df[pos_columns]
  x2 = test_df[neg_columns]

  x1 = x1.values
  x2 = x2.values
  y1 = np.ones((x1.shape[0],), dtype=np.int32)
  y2 = np.zeros((x2.shape[0],), dtype=np.int32)

  return np.vstack((x1,x2)), np.concatenate((y1, y2))


def main(_):
  train_x, train_y, x_columns = load_train_data(FLAGS.emb_dim)
  eval_x, eval_y = load_eval_data(FLAGS.emb_dim)

  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x":train_x},
      y=train_y,
      batch_size=FLAGS.batch_size,
      num_epochs=None,
      shuffle=True
  )

  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": eval_x},
    y=eval_y,
    batch_size=FLAGS.batch_size,
    num_epochs=1,
    shuffle=False
  )

  serving_feature_spec = {"x": tf.FixedLenFeature(dtype=tf.float32, shape=[FLAGS.batch_size*2])}
  serving_input_receiver_fn = (
    tf.estimator.export.build_parsing_serving_input_receiver_fn(
      serving_feature_spec))

  exporter = tf.estimator.BestExporter(
    name="best_exporter",
    serving_input_receiver_fn=serving_input_receiver_fn,
    exports_to_keep=5)

  # tf ver 1.9에서 문제가 있다.
  # 미리 디렉터리를 만들어주지 않으면 에러가 발생. 나중에 수정될 듯...
  e_path = os.path.join(os.path.join(FLAGS.model, "export"),exporter.name)
  pathlib.Path(e_path).mkdir(parents=True, exist_ok=True)

  train_spec = tf.estimator.TrainSpec(input_fn = train_input_fn, max_steps=20000)
  eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=200, exporters=exporter, start_delay_secs=0,
  throttle_secs=1)

  estimator = tf.estimator.DNNClassifier(
    config=tf.estimator.RunConfig(
      model_dir=FLAGS.model, save_summary_steps=100),
    feature_columns=[tf.feature_column.numeric_column(key='x', shape=[200])],
    hidden_units=[32, 32], n_classes=2)

  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument(
      "--train",
      type=str,
      default="data/train_data_w2vec_100K.csv",
      help="path of a train data")

  parser.add_argument(
      "--eval",
      type=str,
      default="data/test_data_w2vec_1M.csv",
      help="path of an evaluation data")

  parser.add_argument(
      "--model",
      type=str,
      default="./my_model",
      help="path of the estimator's model")

  parser.add_argument(
      "--emb_dim",
      type=int,
      default=100,
      help="dimension of word2vector embedding")

  parser.add_argument(
      "--batch_size",
      type=int,
      default=512,
      help="mini-batch size")

  FLAGS, unparsed = parser.parse_known_args()

  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

