# Final-NeuralFBProphet
팀스톤 cpu 데이터를 활용한 이상탐지 및 시계열 예측

## Code Style
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
python black을 활용하여 코드 작성

## Model
#### Prophet
+ 페이스북이 만든 시계열 예측 라이브러리
    + 통계적 지식이 없어도 직관적 파라미터를 통해 모형을 조정할 수 있음
    + 일반적인 경우 기본값만 사용해도 높은 성능을 보여줌
    + Python, R로 사용 가능
+ 관련 자료
    + [모든이들을 위한 FACEBOOK PROPHET PAPER 쉬운 요약정리](https://gorakgarak.tistory.com/1255)
    + [Prophet Paper](https://peerj.com/preprints/3190.pdf)
+ 설치
```
pip install fbprophet
```
#### Neural Prophet
+ 페이스북의 업데이트 된 예측 라이브러리
    + Neural Prophet은 PyTorch를 사용하여 구축되고 시계열 예측을 위해 AR-Net과 같은 딥 러닝 모델을 사용하는 Prophet의 업그레이드된 버전이다.
+ 설치
```
pip install neuralprophet
```
+ 주피터 노트북에 NeuralProphet을 사용할 계획이라면 다음 명령을 사용하여 NeuralProphet의 라이브 버전을 설치할 때 도움이 될 수 있다.
```
pip install neuralprophet[live]
```
