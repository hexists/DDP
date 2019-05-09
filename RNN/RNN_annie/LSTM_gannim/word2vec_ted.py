#-*- coding: utf-8 -*-

import re
import sys

def sent_tokenize(data):
    datas = data.split('.')
    return datas
def word_tokenize(data):
    datas = data.split(' ')
    return datas
with open('./data/ted_en-20160408.xml', 'r') as ff:
    datas = []
    data = ''
    for line in ff.readlines():
        if '<content>' in line:
            data += str.replace(line.strip(), '<content>', '')
        elif '</content>' in line:
            data += str.replace(line.strip(), '</content>', '')
            # content 중간에 등장하는 (Audio), (Laughter) 등의 배경음 부분을 제거.
            data = re.sub(r'\([^)]*\)', '', data)
            datas.append(data)
            data = ''
    docs = []
    for data in datas:
        docs.append(sent_tokenize(data))

normalized_docs= []
for doc in docs:
    norm_sens = [re.sub(r"[^a-z0-9]+", " ", sen.lower()) for sen in doc if len(sen.strip()) > 0]
    for norm_sen in norm_sens:
        normalized_docs.append(word_tokenize(norm_sen))

from gensim.models import Word2Vec

"""
 size = word vector feature value, embedding된 vetor 차원의 값
 window = context windown size
 min_count = 단어 최소 빈도 수 제한
 workers = 학습을 위한 프로세스 수
 sg = 다운 샘플링 비율 
"""
model = Word2Vec(sentences=normalized_docs, size=100, window=5, min_count=5, workers=4, sg=0)
print model.wv.most_similar("man")
model.save("./data/word2vec_ted.model")
