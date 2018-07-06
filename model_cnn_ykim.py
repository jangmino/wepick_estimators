import argparse
import cnn_utils
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.python import debug as tf_debug
import os
import pathlib
import itertools

tf.logging.set_verbosity(tf.logging.INFO)

TARGET_LABELS = [0, 1]

def model_fn(features, labels, mode, params):
  hidden_units = params.hidden_units
  output_layer_size = len(TARGET_LABELS)
  embedding_size = params.embedding_size
  kernel_sizes = params.kernel_sizes
  filter_sizes = params.filter_sizes
  total_filters = np.sum(filter_sizes)

  is_training = True if mode==tf.estimator.ModeKeys.TRAIN else False

  #n_in = tf.shape(embeddings)[1]
  #n_in = 64
  #print("embeddings: {}".format(embeddings)) # (?, MAX_SEQ_LENGH_IN_MINIBATCH, embbeding_size)

  embeddings = tf.layers.dropout(features['w2vecs'], params.dropout_rate, training=is_training)

  # convolutions
  convs = []
  for i in range(len(kernel_sizes)):
    conv = tf.layers.conv1d(inputs=embeddings,
                            filters=filter_sizes[i], kernel_size=kernel_sizes[i],
                            padding='valid', strides=1, activation=tf.sigmoid,
                            name="conv1d_{}".format(i)
                            )
    # print("conv: {}".format(conv)) # (?, MAX_SEQ_LENGH_IN_MINIBATCH - kernel_size, filter_size)

    pool = tf.reduce_mean(input_tensor=conv, axis=1)
    # print("pool: {}".format(pool))
    convs.append(pool)
    # pool = tf.layers.average_pooling1d(conv,
    #                                pool_size=n_in-kernel_sizes[i]+1, strides=1,
    #                                name="average_pooling1d_{}".format(i)
    #                                )
    # print("pool: {}".format(pool))  # (?, 1, filter_size)
    # flattenAvg = tf.layers.flatten(pool, name="flatten_{}".format(i))
    # print("flatten: {}".format(flattenAvg)) # (?, MAX_SEQ_LENGH_IN_MINIBATCH - kernel_size, filter_size)
    #
    # convs.append(flattenAvg)

  input_layer = tf.concat(convs, axis=-1)
  net = tf.reshape(input_layer, [-1, total_filters])

  if hidden_units is not None:
    for h in hidden_units:
      net = tf.layers.dense(net, h, activation=tf.sigmoid)
      net = tf.layers.dropout(net, params.dropout_rate, training=is_training)

  # print("hidden_layer: {}".format(net))

  logits = tf.layers.dense(inputs=net, units=output_layer_size, activation=None)
  # print("logits: {}".format(logits))


  # Provide an estimator spec for `ModeKeys.PREDICT`.
  if mode == tf.estimator.ModeKeys.PREDICT:
    probabilities = tf.nn.softmax(logits)
    predicted_indices = tf.argmax(probabilities, 1)

    # Convert predicted_indices back into strings
    predictions = {
      'class': tf.gather(TARGET_LABELS, predicted_indices),
      'probabilities': probabilities,
#      'sl': features['sl']
    }
    export_outputs = {
      'prediction': tf.estimator.export.PredictOutput(predictions)
    }

    # Provide an estimator spec for `ModeKeys.PREDICT` modes.
    return tf.estimator.EstimatorSpec(mode,
                                      predictions=predictions,
                                      export_outputs=export_outputs)

  # Calculate loss using softmax cross entropy
  loss = tf.losses.sparse_softmax_cross_entropy(
    logits=logits, labels=labels
  )

  tf.summary.scalar('loss', loss)

  if mode == tf.estimator.ModeKeys.TRAIN:
    # Create Optimiser
    optimizer = tf.train.AdamOptimizer(params.learning_rate)

    # Create training operation
    train_op = optimizer.minimize(
      loss=loss, global_step=tf.train.get_global_step())

    # Provide an estimator spec for `ModeKeys.TRAIN` modes.
    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      train_op=train_op)

  if mode == tf.estimator.ModeKeys.EVAL:
    probabilities = tf.nn.softmax(logits)
    predicted_indices = tf.argmax(probabilities, 1)

    # Return accuracy and area under ROC curve metrics
    labels_one_hot = tf.one_hot(
      labels,
      depth=len(TARGET_LABELS),
      on_value=True,
      off_value=False,
      dtype=tf.bool
    )

    eval_metric_ops = {
      'accuracy': tf.metrics.accuracy(labels, predicted_indices),
      'auroc': tf.metrics.auc(labels_one_hot, probabilities)
    }

    # Provide an estimator spec for `ModeKeys.EVAL` modes.
    return tf.estimator.EstimatorSpec(mode,
                                      loss=loss,
                                      eval_metric_ops=eval_metric_ops)

def serving_input_fn(hparams):
  receiver_tensor = {
    'x': tf.placeholder(tf.string, [None]),
  }

  features = {
    'w2vecs': tf.expand_dims(tf.reshape(tf.decode_raw(receiver_tensor['x'], tf.float32), [-1, hparams.embedding_size]),0)
    for key, tensor in receiver_tensor.items()
  }

  return tf.estimator.export.ServingInputReceiver(
    features, receiver_tensor)

# def serving_input_fn(hparams):
#   receiver_tensor = {
#     'x': tf.placeholder(tf.string, [None]),
#   }
#
#   features = {'w2vecs': tf.expand_dims(tf.reshape(tf.decode_raw(receiver_tensor['x'], tf.float32), [-1, hparams.embedding_size]),0)}
#
#   return tf.estimator.export.ServingInputReceiver(
#     features, receiver_tensor)


def create_estimator(run_config, hparams):
  estimator = tf.estimator.Estimator(model_fn=model_fn,
                                     params=hparams,
                                     config=run_config)

  print("")
  print("Estimator Type: {}".format(type(estimator)))
  print("")

  return estimator


# TRAIN_DATA_FILES_PATTERN = r'd:\WMIND\temp\train_data\data-*.tfrecords'
# VALID_DATA_FILES_PATTERN = r'd:\WMIND\temp\test_data\evaluation_small\data-*.tfrecords'
# TRAIN_SIZE = 17000000
# NUM_EPOCHS = 10
# MODEL_NAME = 'wepick-cnn-ykim-01'
TRAIN_DATA_FILES_PATTERN = r'd:\WMIND\temp\train_data\data-00?1.tfrecords'
# VALID_DATA_FILES_PATTERN = r'd:\WMIND\temp\train_data\data-0001.tfrecords'
VALID_DATA_FILES_PATTERN = r'd:\WMIND\temp\test_data\evaluation_small\data-00??.tfrecords'
TRAIN_SIZE = 320000 * 5
NUM_EPOCHS = 10
BATCH_SIZE = 1024
EVAL_AFTER_SEC = 120
TOTAL_STEPS = int((TRAIN_SIZE/BATCH_SIZE)*NUM_EPOCHS)
MODEL_NAME = 'wepick-cnn-ykim-small'

## 최대 시퀀스 길이는 64

def main(unused_argv):
  hparams = tf.contrib.training.HParams(
    num_epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    embedding_size=100,
    hidden_units=[64, 32],  # [8],
    dropout_rate=0.3,
    max_steps=TOTAL_STEPS,
    kernel_sizes=[32, 32, 16, 16,  8,  8,  4,  4,  2,  2],
    filter_sizes=[16, 16, 16, 16, 16, 16, 16, 16, 16, 16],
    learning_rate=0.001,
    log_step_count_steps=2000,
    max_sequence_len=64
  )

  model_dir = 'trained_models/{}'.format(MODEL_NAME)

  tf.logging.info("hparams:{}".format(hparams))

  run_config = tf.estimator.RunConfig(
    log_step_count_steps=hparams.log_step_count_steps,
    tf_random_seed=19830610,
    model_dir=model_dir
  )

  train_spec = tf.estimator.TrainSpec(
    input_fn = lambda: cnn_utils.tfrecords_input_fn(
      TRAIN_DATA_FILES_PATTERN,
      mode=tf.estimator.ModeKeys.TRAIN,
      num_epochs=hparams.num_epochs,
      batch_size=hparams.batch_size,
      max_history_len=hparams.max_sequence_len
    ),
    max_steps=hparams.max_steps,
    hooks=None
  )

  eval_spec = tf.estimator.EvalSpec(
    input_fn=lambda: cnn_utils.tfrecords_input_fn(
      VALID_DATA_FILES_PATTERN,

      mode=tf.estimator.ModeKeys.EVAL,
      batch_size=hparams.batch_size,
      max_history_len=hparams.max_sequence_len
    ),
    exporters=[
      tf.estimator.BestExporter(
        name="best_exporter",
        serving_input_receiver_fn=lambda: serving_input_fn(hparams),
        exports_to_keep=3,
        as_text=True),
      tf.estimator.LatestExporter(
        name="latest_exporter",
        serving_input_receiver_fn=lambda: serving_input_fn(hparams),
        exports_to_keep=1,
        as_text=True)
    ],
    steps=None,
    throttle_secs=EVAL_AFTER_SEC
  )

  # tf ver 1.9에서 문제가 있다.
  # 미리 디렉터리를 만들어주지 않으면 에러가 발생. 나중에 수정될 듯...
  e_path = os.path.join(os.path.join(model_dir, "export"), "best_exporter")
  pathlib.Path(e_path).mkdir(parents=True, exist_ok=True)
  e_path = os.path.join(os.path.join(model_dir, "export"), "latest_exporter")
  pathlib.Path(e_path).mkdir(parents=True, exist_ok=True)

  estimator = create_estimator(run_config, hparams)
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

  # predictions = list(
  #   itertools.islice(
  #     estimator.predict(
  #       input_fn=lambda: cnn_utils.tfrecords_input_fn(
  #         TRAIN_DATA_FILES_PATTERN,
  #         mode=tf.estimator.ModeKeys.PREDICT,
  #         batch_size=32,
  #         max_history_len=64
  #       )
  #     ), 32
  #   )
  # )
  # print(predictions)


if __name__ == "__main__":
  # tf.enable_eager_execution()
  tf.app.run()
