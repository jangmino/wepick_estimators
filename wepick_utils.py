import time
import json
import elasticsearch
import csv
import glob
import os
from datetime import timezone, timedelta, datetime
import pandas as pd
import numpy as np
import tensorflow as tf
from operator import itemgetter


def es_search_dids_for_user(es, user_id, day_limit, gte_slot=1, ignore_consecutives=False):
  """
  user_id의 모든 v 가져오기
  day_limit 이전 것만 가져온다.
  return
  - 1st: v의 set
  - 2nd: 확장 정보 (v, rgtime, slot)
  """
  res = es.search(index='wepick_seq',
                  body={
                    "query": {
                      "bool": {
                        "must": {
                          "term": {"u": user_id}
                        },
                        "filter": {
                          "range": {
                            "rgtime": {
                              "lt": day_limit
                            }
                          }
                        }
                      }
                    },
                    "size": 64,
                    "sort": {"rgtime": "desc"}
                  }
                  )
  if res['hits']['total'] > 0:
    until_dt = pd.to_datetime(day_limit).to_pydatetime()
    filtered = []
    prev_v = None
    for hit in res['hits']['hits']:
      if ignore_consecutives == False or (prev_v is not None and prev_v != hit['_source']['v']):
        filtered.append((hit['_source']['v'], hit['_source']['rgtime'], hit['_source']['slot']))
      prev_v = hit['_source']['v']
    return set(map(lambda x: x[0], filtered)), filtered
  return None, None


def es_gather_word2vec_dids(es, dids):
    """
    dids로부터, word2vec을 모은다.
    - 모아서, average pooling 실시
    return
    - vector normalized by L2-norm
    """
    res = es.search(index='deal_word2vec',
                    body={
                      'from': 0, 'size': len(dids),
                      "_source": ["values"],
                      'query': {
                        'ids': {'values': dids}
                      }
                    }
                    )
    mat = []
    for hit in res['hits']['hits']:
      vec = np.array(hit['_source']['values'])
      if len(vec) > 0:
        mat.append(vec)
    vec = np.mean(np.vstack(mat), axis=0)
    vec /= np.sqrt(np.sum(vec ** 2))
    return vec


def es_gather_word2vec_wepick(es, dids):
  """
  dids로부터, word2vec을 모은다.
  return
  - dids: unit-length w2v (normalized by L2-norm)
  """
  res = es.search(index='deal_word2vec',
                  body={
                    'from': 0, 'size': len(dids),
                    "_source": ["values", "v"],
                    'query': {
                      'ids': {'values': dids}
                    }
                  }
                  )
  dic = {}
  for hit in res['hits']['hits']:
    did = hit['_source']['v']
    vec = np.array(hit['_source']['values'])
    if len(vec) > 0:
      vec /= np.sqrt(np.sum(vec ** 2))
      dic[did] = vec
  return dic


def es_read_wepick_setting(es, dt, start_slot=20):
  """
  위픽 세팅 로딩
  """
  res = es.search(index='wepick_setting_ext',
                  body={
                    'query': {
                      'term': {'dt': dt}
                    }
                  }
                  )
  if res['hits']['total'] > 0:
    dic = {}
    vec = []
    for s in res['hits']['hits'][0]['_source']['settings']:
      if s['slot'] >= start_slot:
        dic[s['slot']] = s['did']
        vec.append(s['did'])
    return vec, dic
  return None, None

def es_scan_extra_by_dids(es, dids):
  """
  dids로부터, mn, tn1를 가져온다.
  """
  res = es.search(index='dealinfos',
                  body={
                    'from': 0, 'size': len(dids),
                    "_source": ["mn", "tn1", "did"],
                    'query': {
                      'ids': {'values': dids}
                    }
                  }
                  )
  dic = {}
  for hit in res['hits']['hits']:
    dic[hit['_source']['did']] = (hit['_source']['mn'], hit['_source']['tn1'])
  return dic

def make_tf_serving_request(user_profile, deal_profile_dic):
  """
  서빙을 위한 위픽 예측용 데이터 생성
  example = user_profile + wepick_deal_1
  ...
  ...

  :param user_profile:
  :param deal_profile_dic:
  :return:
  """
  examples = []
  for did, vec in deal_profile_dic.items():
    x = np.concatenate((user_profile, vec))
    feature = {"x": tf.train.Feature(float_list=tf.train.FloatList(value=x))}
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    examples.append(example.SerializeToString())
  return examples


def sort_predictions(predictions, deal_profile_dic):
  scores = list(zip(deal_profile_dic.keys(), predictions['scores'][:, 1]))
  return sorted(scores, key=itemgetter(1), reverse=True)

def get_refined_scores(scores, extra_dic):
  refined_scores = []
  for did, score in scores:
    if did in extra_dic:
      refined_scores.append((score, did, extra_dic[did][0], extra_dic[did][1]))
    else:
      refined_scores.append((score, did, "", ""))
  return refined_scores


def print_result(out, wepick_slot_dic):
  for s, did, title, cate in out:
    org_slot = wepick_slot_dic[did] if did in wepick_slot_dic else -1
    print((s, did, title, org_slot, cate))


# if __name__ == "__main__":
#   # ### twiceSpark1
#   # es_url = '10.102.50.47:9200'
#   # es = elasticsearch.Elasticsearch(es_url)
#   #
#   # # 2018-04-11 09 시의 위픽 세팅 로딩
#   # wepick_setting, wepick_dic = es_read_wepick_setting(es, '2018-04-11 21')
#   # wepick_slot_dic = dict(zip(wepick_dic.values(), wepick_dic.keys()))
#   # extra_dic = es_scan_extra_by_dids(es, wepick_setting)
#   #
#   # # 위픽 세팅에 따른 딜들에 대한 deal_profile을 생성
#   # deal_profile_dic = es_gather_word2vec_wepick(es, wepick_setting)
#   #
#   # deals_user_viewed, ex = es_search_dids_for_user(es, 1000007, '2018-04-11')
#   # user_profile = es_gather_word2vec_dids(es, list(deals_user_viewed))
#   #
#   # export_dir = r'c:\Users\wmp\TensorFlow\wepick_estimators\my_model\export\best_exporter\\1529993143\\'
#   # predict_fn = tf.contrib.predictor.from_saved_model(export_dir)
#   #
#   # examples = make_tf_serving_request(user_profile, deal_profile_dic)
#   # predictions = predict_fn({'inputs': examples})
#   #
#   # result = sort_predictions(predictions, deal_profile_dic)
#   #
#   # print_result(get_refined_scores(result, extra_dic), wepick_slot_dic)
#
