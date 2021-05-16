# Final-NeuralFBProphet
팀스톤 cpu 데이터를 활용한 이상탐지 및 시계열 예측

## Code Style
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
python black을 활용하여 코드 작성


## Data
+ 15day_0201_0215.csv
    + cpu 사용량을 2021년 2월 1일부터 2021년 2월 15일치를 2초동안 log로 남김
+ ontune2016.csv
    + 팀스톤에서 자체 개발한 데이터베이스 서버의 log기록을 10분 단위로 남김
+ vcenter.csv
    + 팀스톤에서 자체 개발한 데이터베이스 서버의 log기록을 10분 단위로 남김

## Model
#### Prophet
+ 페이스북이 만든 시계열 예측 라이브러리
    + 통계적 지식이 없어도 직관적 파라미터를 통해 모형을 조정할 수 있음
    + 일반적인 경우 기본값만 사용해도 높은 성능을 보여줌
    + Python, R로 사용 가능
+ 관련 자료
    + [모든이들을 위한 FACEBOOK PROPHET PAPER 쉬운 요약정리](https://gorakgarak.tistory.com/1255)
    + [Prophet Paper](https://peerj.com/preprints/3190.pdf)

#### Neural Prophet
+ 페이스북의 업데이트 된 예측 라이브러리
    + Neural Prophet은 PyTorch를 사용하여 구축되고 시계열 예측을 위해 AR-Net과 같은 딥 러닝 모델을 사용하는 Prophet의 업그레이드된 버전이다.
#### LSTM
+  LSTM은 RNN의 히든 state에 cell-state를 추가한 구조
    + 바닐라 RNN에 비해 LSTM 하나의 구조에도 네 개의 뉴럴 네트워크가 들어가 있지만, 실제로 TensorFlow 를 이용하서 사용할 땐 간단함
    + 입력과 출력을 정해주고, 초기화만 잘 시켜주면, 텐서플로우에서 LSTM 모듈을 사용할 수 있음

## Hyper Parameter Tuning
+ Bayesian TPE 방식
    + Bayesian 접근 방식은 그리드 검색 및 임의 검색 보다 훨씬 더 효율적일 수 있습니다. 따라서 TPE (Parzen 추정) 알고리즘 알고리즘을 사용 하면 더 많은 하이퍼 매개 변수 및 더 큰 범위를 탐색할 수 있다. 도메인 정보를 사용 하여 검색 도메인을 제한 하면 튜닝을 최적화 하고 더 나은 결과를 얻을 수 있다.
    + 따라서 이 프로젝트의 핵심적인 요소는 하이퍼파라미터 튜닝에 관련되어있다고 보고
    하이퍼파라미터 튜닝을 AutoML방식으로 사용할 것을 제안했다.
    + 특정 데이터에서는 높은 성능을 발휘하였다.

## BenchMark
+ 평가지표는 RMSE를 활용

|DataSet|LSTM|Prophet|Neural-Prophet|
|-------|:---|:------|:-------------|
|15day|**4.137315123425911**|5.740843522277815|11.500915205148136|
|OnTune|**0.2803152442875646**|1.7460619248959035|20.68509903152376|
|vcenter|3.819914152716074|**0.6587776491484609**|14.284093776822472|
