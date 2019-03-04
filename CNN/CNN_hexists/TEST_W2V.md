# CNN Test with word2vec

```
DennyBritz 코드를 base로 성능 향상을 위해 word2vec을 추가해 본 테스트
```

## ref
- [https://gist.github.com/j314erre/b7c97580a660ead82022625ff7a644d8](https://gist.github.com/j314erre/b7c97580a660ead82022625ff7a644d8)
- [https://stats.stackexchange.com/a/276750](https://stats.stackexchange.com/a/276750)

## Test

- 위 ref에 있는 코드를 바탕으로 gensim으로 빌드된 word2vec을 붙여봄
- gensim으로 빌드된 word2vec은 아래 주소에 존재
  - [https://code.google.com/archive/p/word2vec/](https://code.google.com/archive/p/word2vec/)
- model load

  ```
  import gensim
  model = gensim.models.KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)
  ```
- 기존 코드에서 gensim을 라인 단위로 읽어서 embedding을 불러오는 부분을 gensim 모델을 load하는 것으로 변경
  - 그 다음에 학습 데이터의 vocab 정보를 이용해서 initW에 저장 후, 다시 W에 assign하는 방법으로 word2vec을 사용
  ```
  wv_model = gensim.models.KeyedVectors.load_word2vec_format(FLAGS.word2vec, binary=True)
  vocab_dict = vocab_processor.vocabulary_._mapping
  for word, i in vocab_dict.items():
      if word in wv_model.vocab:
          initW[i] = wv_model[word]
  ```

## Result

- 기존 대비 성능이 어떻게 변했는지를 보면,
  - 기존에 최고 성능은 dev set acc 75%
  - word2vec을 추가한 뒤 최고 성능은 dev set acc 78.99%
  - 기존 대비 약 4% 향상

## 추가 실험

- word2vec을 cnn class에 전달하여, get_variable로 할당해서 사용하도록 해봤습니다.
- 사용 방법이 다르고, 결과에는 차이가 없습니다.
- code
  - text_cnn_w2v_v2.py
  - train_w2v_v2.py
  
  ```
  # Embedding layer
  with tf.device('/cpu:0'), tf.name_scope("embedding"), tf.variable_scope("embedding"):
      self.W = tf.get_variable(name="W", initializer=tf.constant_initializer(self.initMatrix), shape=self.initMatrix.shape, trainable=False)
  ```
