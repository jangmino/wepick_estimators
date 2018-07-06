import wepick_utils as wutil
import elasticsearch
import tensorflow as tf
import sys
import argparse
import pickle
import numpy as np
from operator import itemgetter

def predict_by_base_model(es, deal_profile_dic, extra_dic, wepick_slot_dic):
  deals_user_viewed, ex = wutil.es_search_dids_for_user(es, FLAGS.mid, FLAGS.dt.split(' ')[0])
  user_profile = wutil.es_gather_word2vec_dids(es, list(deals_user_viewed))

  export_dir = FLAGS.model_path
  predict_fn = tf.contrib.predictor.from_saved_model(export_dir)

  examples = wutil.make_tf_serving_request(user_profile, deal_profile_dic)
  predictions = predict_fn({'inputs': examples})

  result = wutil.sort_predictions(predictions, deal_profile_dic)

  wutil.print_result(wutil.get_refined_scores(result, extra_dic), wepick_slot_dic)


def predict_by_cnn_ykim(es, deal_profile_dic, extra_dic, wepick_slot_dic):
  with open(FLAGS.w2vec_dic, 'rb') as f:
    deal_word2vec_dic = pickle.load(f)

  deal_word2vec_bytes = {}
  for did, vec in deal_word2vec_dic.items():
    deal_word2vec_bytes[did] = vec.tobytes()

  deals_user_viewed, ex = wutil.es_search_dids_for_user(es, FLAGS.mid, FLAGS.dt.split(' ')[0], ignore_consecutives=True)
  examples=[]
  export_dir = FLAGS.model_path
  predict_fn = tf.contrib.predictor.from_saved_model(export_dir, signature_def_key="prediction")

  scores = []
  history = list(map(lambda x: x[0], ex))
  for wepick_did in deal_profile_dic.keys():
    dids = [wepick_did] + history
    contents = []
    for did in dids:
      if did in deal_word2vec_dic:
        contents.append(deal_word2vec_dic[did])
    contents = np.array(contents, dtype=np.float32)

    ojm = predict_fn({"x": [contents.tobytes()]})
    scores.append((wepick_did, ojm['probabilities'][0, 1]))

  scores = sorted(scores, key=itemgetter(1), reverse=True)
  wutil.print_result(wutil.get_refined_scores(scores, extra_dic), wepick_slot_dic)


def main(_):
  es_url = FLAGS.esaddr
  es = elasticsearch.Elasticsearch(es_url)

  # 2018-04-11 09 시의 위픽 세팅 로딩
  wepick_setting, wepick_dic = wutil.es_read_wepick_setting(es, FLAGS.dt)
  wepick_slot_dic = dict(zip(wepick_dic.values(), wepick_dic.keys()))
  extra_dic = wutil.es_scan_extra_by_dids(es, wepick_setting)

  # 위픽 세팅에 따른 딜들에 대한 deal_profile을 생성
  deal_profile_dic = wutil.es_gather_word2vec_wepick(es, wepick_setting)

  if FLAGS.strategy == "base_model":
    predict_by_base_model(es, deal_profile_dic, extra_dic, wepick_slot_dic)
  elif FLAGS.strategy == "cnn_ykim":
    predict_by_cnn_ykim(es, deal_profile_dic, extra_dic, wepick_slot_dic)
  else:
    print("{} is not matched...".format(FLAGS.strategy))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument(
      "--model_path",
      type=str,
      default=r"./trained_models/wepick-cnn-ykim-small\export\latest_exporter\\1530862658\\",
      help="path of a train data")

  parser.add_argument(
      "--esaddr",
      type=str,
      default="10.102.50.47:9200",
      help="address of elastic search")

  parser.add_argument(
      "--mid",
      type=int,
      default=1000007,
      help="user mid")

  parser.add_argument(
      "--dt",
      type=str,
      default="2018-04-11 21",
      help="dt of wepick setting for prediction")

  parser.add_argument(
      "--strategy",
      type=str,
      default="cnn_ykim",
      help="[base_model, cnn_ykim]")

  parser.add_argument(
      "--w2vec_dic",
      type=str,
      default=r'd:\WMIND\temp\deal_to_w2vec.pkl',
      help="path of deal-to-w2vec dictionary")


  FLAGS, unparsed = parser.parse_known_args()

  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
