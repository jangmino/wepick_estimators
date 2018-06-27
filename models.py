import argparse
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.python import debug as tf_debug

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

  train_df = pd.read_csv(r'd:\WMIND\temp\train_data_w2vec_100K.csv', header=None, names=name_header, dtype=dtypes,
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

  test_df = pd.read_csv(r'd:\WMIND\temp\test_data_w2vec_1M.csv', header=None, names=name_header, dtype=test_dtypes,
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

def my_model(features, labels, mode, params):
  #net = tf.feature_column.input_layer(features, params['feature_columns'])
  net = features["x"]
  for units in params['hidden_units']:
    net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
    net = tf.layers.batch_normalization(net, training=True if mode == tf.estimator.ModeKeys.TRAIN else False)

  # Compute logits (1 per class).
  logits = tf.layers.dense(net, 2, activation=None)

  # Compute predictions.
  predicted_classes = tf.argmax(logits, 1)
  # if mode == tf.estimator.ModeKeys.PREDICT:
  #   predictions = {
  #     'class_ids': predicted_classes[:, tf.newaxis],
  #     'probabilities': tf.nn.softmax(logits),
  #     'logits': logits,
  #   }
  #   return tf.estimator.EstimatorSpec(mode, predictions=predictions)

  # Compute loss.
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Compute evaluation metrics.
  accuracy = tf.metrics.accuracy(labels=labels,
                                 predictions=predicted_classes,
                                 name='acc_op')
  auc = tf.metrics.auc(labels=labels,
                                 predictions=predicted_classes,
                                 name='auc')
  eval_metrics = {'auc': auc}

  if mode == tf.estimator.ModeKeys.EVAL:
    return tf.estimator.EstimatorSpec(
      mode, loss=loss, eval_metric_ops=eval_metrics)

  # Create training op.
  assert mode == tf.estimator.ModeKeys.TRAIN

  #optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)

  tensors_to_log = {'batch_accuracy': accuracy[1],
                    'logits': logits,
                    'label': labels}
  logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=500)

  optimizer = tf.train.AdamOptimizer()
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
  return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

def main(unused_argv):
  train_x, train_y, x_columns = load_train_data()
  eval_x, eval_y = load_eval_data()

  # Create the Estimator
#  run_config = tf.contrib.learn.RunConfig()
#  run_config = run_config.replace(save_summary_steps=20)
  my_classifier = tf.estimator.Estimator(
      model_fn=my_model, model_dir="./model",
      params = {
        'feature_columns':x_columns,
        'hidden_units':[32,32],
        'n_classes': 1
      }
#      config=run_config
  )

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x":train_x},
      y=train_y,
      batch_size=512,
      num_epochs=None,
      shuffle=True
  )
  # for i in range(100):
  #   my_classifier.train(
  #     input_fn=train_input_fn,
  #     steps=100
  #     )
  #
  #   eval_input_fn = tf.estimator.inputs.numpy_input_fn(
  #     x={"x": eval_x},
  #     y=eval_y,
  #     batch_size=512,
  #     num_epochs=1,
  #     shuffle=False
  #   )
  #
  #   result = my_classifier.evaluate(input_fn=eval_input_fn)
  #   print("i: {}, gs:{}, loss: {}, auc: {}".format(i, result['global_step'], result['loss'], result['auc']))

  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": eval_x},
    y=eval_y,
    batch_size=512,
    num_epochs=1,
    shuffle=False
  )

  serving_feature_spec = {"x": tf.FixedLenFeature(dtype=tf.float32, shape=[200])}
  serving_input_receiver_fn = (
    tf.estimator.export.build_parsing_serving_input_receiver_fn(
      serving_feature_spec))

  exporter = tf.estimator.BestExporter(
    name="best_exporter",
    serving_input_receiver_fn=serving_input_receiver_fn,
    exports_to_keep=5)

  train_spec = tf.estimator.TrainSpec(input_fn = train_input_fn, max_steps=20000)
  eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps = 200, exporters=exporter, start_delay_secs=0,
  throttle_secs=1)

  estimator = tf.estimator.DNNClassifier(
    config=tf.estimator.RunConfig(
      model_dir='./my_model', save_summary_steps=100),
    feature_columns=[tf.feature_column.numeric_column(key='x', shape=[200])],
    hidden_units=[32, 32], n_classes=2)

  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

if __name__ == "__main__":
  tf.app.run()
