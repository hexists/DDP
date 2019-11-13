Simple BERT using TF2.0
-----

- [towarddatascience](https://towardsdatascience.com/simple-bert-using-tensorflow-2-0-132cb19e9b22)에 있는 설명을 보고 따라해봤습니다.

### 요약

- tf_hub에 있는 BERT model을 이용하는 단순한 예제입니다.
- tf_hub에 있는 BERT를 불러오고, tokenizer 생성하고, 그걸로 입력 문장(sentence)에 대해서 pool\_embs, all\_embs를 구합니다.

### 준비

- tensorflow-gpu, tensorflow-hub를 설치해야 합니다.
- tensorflow-gpu 2.0.0 이상, tensorflow-hub 0.7.0 이상 설치
- 위 환경을 virtualenv를 이용해서 가상 환경에 구성했고, 이를 requirements.txt로 저장했습니다.

  ```
  $ virtualenv python3 -m .venv
  $ source .venv/bin/activate
  $ pip install --no-cache-dir requirements.txt
  ```
- tokenizer 사용을 위해서는 bert repo를 clone 해야 합니다.
- clone 후 path를 맞춰주기 위해, symbolic link를 만듭니다.

  ```
  $ git clone https://github.com/tensorflow/models.git
  $ ln -sf models/official/nlp/bert .
  ```
- tensorflow-gpu 사용으로 인한 cuda path 설정은 각자 cuda9.0 lib가 있는 path를 추가해서 사용하면 됩니다.

### 실행

```
$ python simple_bert_using_tf2.py

...

s       : This is a nice sentence.
stokens : ['[CLS]', 'this', 'is', 'a', 'nice', 'sentence', '.', '[SEP]']

pool_embs.shape:  (1, 768)
all_embs.shape :  (1, 128, 768)
cls_embs.shape :  (1, 768)

cosine_similarity(pool_embs, cls_embs): [[0.02757266]]
```


### 해석

```
s       : 입력 문장
stokens : BERT tokenizer를 이용해서 tokenize

pool_embs.shape: sentence-level embedding
all_embs.shape : token embeddings
cls_embs.shape : contextualised embedding, [CLS] embedding

cosine_similarity(pool_embs, cls_embs): 두 임베딩의 cos similiarity 결과

cosine similarity를 계산해 본 이유는 cls_embs([CLS])와 pool_embs가 의미적으로는 모두 sentence-level embedding인데, 두 embedding의 결과가 차이 난다는 점을 언급하고 있습니다.
두 임베딩을 어떻게 바라봐야 하는지는 좀 더 공부해봐야 할 것 같습니다.
```
