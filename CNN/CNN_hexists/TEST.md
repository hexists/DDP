# CNN Test

```
DennyBritz 코드를 base로 성능 향상을 위해 여러가지 테스트를 해 본 내용
```

## ref
- [https://github.com/dennybritz/cnn-text-classification-tf](https://github.com/dennybritz/cnn-text-classification-tf)


## Test

1) filter_sizes에 2를 추가

```
3,4,5로 되어 있던 것에 2를 추가하여 2,3,4,5 필터를 사용
의도는 bigram 정보도 유의미할 것이라 생각하여 추가
```
  - 성능 변화(dev 기준): 70% => 75% 상승

2) embedding_dim 변경

```
128로 되어 있던 기본값을 256으로도 증가시켜보고, 64로도 낮춰 테스트 진행
```
  - 성능 변화(dev 기준): 75% 에서 변화없음
  - embedding_dim의 영향이 현 테스트에서는 없어 보임
  - 테스트 데이터가 작아서 그럴 수도 있을 것 같음

3) num_epochs 변경

```
200으로 되어 있지만,
step(전체 개수 / batch_size * num_epochs)이 2000 부터
overfitting 되는 것을 보고, 현 테스트에서는 의미 없다고 생각하여 변경
```

4) num_filters 변경

```
overfitting이 좀 덜 되게 해보려고 128 => 256으로 변경
나중에 확인한 내용이지만, 파라미터가 더 많아져서 더 overfitting 됨
```

5) hidden layer 추가

```
CNN => FC layer 사이에 hidden layer를 추가하여 좀 더 학습이 잘되게 해보려고 했음
나중에 확인한 내용이지만, 파라미터가 더 많아져서 더 overfitting 됨
```

6) L2 regularization 설정

```
모든 파라미터 제곱 만큼의 크기를 목적 함수에 제약을 거는 방식으로 구현된다.
다시말해, 가중치 벡터 w가 있을때, 목적 함수에 1/2λw2를 더한다(여가서 lambda는 regulrization의 강도를 의미).
1/2 부분이 항상 존재하는데 이는 앞서 본 regularization 값을 w로 미분했을 때 2λw가 아닌 λw의 값을 갖도록 하기 위함이다.
L2 reguralization은 큰 값이 많이 존재하는 가중치에 제약을 주고, 가중치 값을 가능한 널리 퍼지도록 하는 효과를 주는 것으로 볼 수 있다. 
```
  - [http://aikorea.org/cs231n/neural-networks-2-kr/](http://aikorea.org/cs231n/neural-networks-2-kr/)

7) dropout_keeprob 조정

```
- 0.5로 설정된 값을 늘려보기도 줄여보기도 해 봄
- 여기서 한가지 주의할 점은 본 프로그램에서 dropout 파라미터는 keep prob에 관한 것
- 값이 높을수록 dropout을 덜 한다는 의미
- 결과
  - 기본으로 설정된 0.5일 때보다 0.25로 dropout_keeprob 비율을 줄였을 때 성능이 미세하게 향상된다.
  - 덧, loss 그래프도 좀 더 이쁘게 보인다.
```
