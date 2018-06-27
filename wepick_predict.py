import wepick_utils as wutil
import elasticsearch
import tensorflow as tf
import sys
import argparse

def main(_):
  es_url = FLAGS.esaddr
  es = elasticsearch.Elasticsearch(es_url)

  # 2018-04-11 09 시의 위픽 세팅 로딩
  wepick_setting, wepick_dic = wutil.es_read_wepick_setting(es, FLAGS.dt)
  wepick_slot_dic = dict(zip(wepick_dic.values(), wepick_dic.keys()))
  extra_dic = wutil.es_scan_extra_by_dids(es, wepick_setting)

  # 위픽 세팅에 따른 딜들에 대한 deal_profile을 생성
  deal_profile_dic = wutil.es_gather_word2vec_wepick(es, wepick_setting)

  deals_user_viewed, ex = wutil.es_search_dids_for_user(es, FLAGS.mid, FLAGS.dt.split(' ')[0])
  user_profile = wutil.es_gather_word2vec_dids(es, list(deals_user_viewed))

  export_dir = FLAGS.model_path
  predict_fn = tf.contrib.predictor.from_saved_model(export_dir)

  examples = wutil.make_tf_serving_request(user_profile, deal_profile_dic)
  predictions = predict_fn({'inputs': examples})

  result = wutil.sort_predictions(predictions, deal_profile_dic)

  wutil.print_result(wutil.get_refined_scores(result, extra_dic), wepick_slot_dic)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument(
      "--model_path",
      type=str,
      default=r"./my_model\export\best_exporter\\1529993143\\",
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


  FLAGS, unparsed = parser.parse_known_args()

  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
