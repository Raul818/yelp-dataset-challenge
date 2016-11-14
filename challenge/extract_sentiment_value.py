#!/usr/bin/env python3

from corenlp_pywrap import pywrap

import threading
import pandas as pd
import sys
import time

lock = threading.Lock()

def write_log(file_prefix, id_str, err):
    lock.acquire()
    try:
        with open('../out/' + file_prefix + '_sentiment.log', mode='a') as f:
            f.write('Failed to get sentiment value for: ' + id_str + ' (Error: ' + str(err[0]) + ' ' + str(err[1]) + ')\n')
            f.flush()
    finally:
        lock.release()

def gen_id_str(data_json, keys):
    key_dict = {}
    for k in keys:
        key_dict[k] = data_json[k]
    return str(key_dict)

def get_sentiment_value(conn, text):
    out = conn.basic(text, out_format='json').json()

    count_of_sentiment_values = {}
    for sentence in out['sentences']:
        sentiment_value = int(sentence['sentimentValue']) - 2    # convert value scale from [0, 4] to [-2, 2]
        if sentiment_value in count_of_sentiment_values:
            count_of_sentiment_values[sentiment_value] += 1
        else:
            count_of_sentiment_values[sentiment_value] = 1

    summ = 0
    for sv in count_of_sentiment_values:
        summ += sv * count_of_sentiment_values[sv]

    return summ * 1.0 / len(out['sentences'])

# Create the connection to the CoreNLP server
conn = pywrap.CoreNLP(url='http://128.114.58.194:9000', annotator_list=['sentiment'])

def process_chunk(f, config, chunk, chunk_index):
    start_index = config['chunk_size'] * chunk_index
    for index in range(start_index, start_index + len(chunk)):
        data_json = pd.read_json(chunk[0][index], typ='series')

        # remove the text content and replace with the sentiment value
        try:
            data_json['sentiment_value'] = get_sentiment_value(conn, data_json['text'])
        except:
            id_str = gen_id_str(data_json, config['id_keys'])
            err = sys.exc_info()
            write_log(config['file_prefix'], id_str, err)
            print('[LOG] Error process item of: ' + id_str)

            data_json['sentiment_value'] = 3     # 3 is the error mark

        data_json.drop(labels='text', inplace=True, errors='ignore')

        lock.acquire()
        try:
            f.write(data_json.to_json() + '\n')
            f.flush()
        finally:
            lock.release()

    print('## Completed chunk ' + str(chunk_index))

process_file_config = {'file_prefix': 'yelp_academic_dataset_review', 'chunk_size': 206543, 'id_keys': ['review_id']}
# process_file_config = {'file_prefix': 'yelp_academic_dataset_tip', 'chunk_size': 58991, 'id_keys': ['user_id', 'business_id']}

reader = pd.read_table('../../dataset/academic/' + process_file_config['file_prefix'] + '.json', header=None, chunksize=process_file_config['chunk_size'])
with open('../out/' + process_file_config['file_prefix'] + '_sentiment.json', mode='w') as f:
    chunk_index = 0
    threads = []
    for chunk in reader:
        t = threading.Thread(target=process_chunk, args=(f, process_file_config, chunk, chunk_index))
        t.start()
        threads.append(t)
        time.sleep(2)    # wait 2 seconds before starting another thread
        chunk_index += 1

    for t in threads:
        t.join()
