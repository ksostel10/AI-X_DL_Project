# AI-X_DL_Project
24-2 한양대학교 AI-X 딥러닝 프로젝트

# Contents
Members

I. Proposal

II. Datasets

III. Methodology

IV. Evaluation & Analysis

V. Related Work

VI. Conclusion

# Members

김영일/ 경제금융학과 / il0906@naver.com

Task : 

정재희/  / 

Task : 

양형석/  /

Task : 

# I. Proposal

### Motivation

 지금 우리 사회에서 사람들이 가장 많이 향유하고 있는 취미를 하나만 선정하자면 단연코 유튜브 시청일 것이다. 많은 사람들이 영상으로 된 컨텐츠, 특히 유튜브 쇼츠의 등장으로 짧지만 큰 임팩트를 주는 영상을 선호하는 방향으로 흘러가고 있다. 이러한 사회적 상황에 맞게, 우리는 긴 영상에서 highlight 부분을 자동으로 추출해 줄 수 있는 딥러닝 모델을 구현하려 한다. 

 그 중에서도 우리는 축구 경기 영상을 선택하게 되었다. 축구는 전반전과 후반전 각 45분에 추가시간까지 부여되기 때문에 총 영상의 길이가 100분 이상인 경우가 대부분이다. 이 중에서 Goal(골), Penalty(반칙), Shots on target(유효슈팅), Shots off target(유효슈팅이 아닌 슈팅) 등 축구 경기 내에서 주목할 만한 부분들만 추출하여 영상으로 제작할 수 있다면 짧은 시간만으로 대부분의 축구 경기 내용을 파악하면서 더 재미있게 축구 영상을 시청할 수 있을 것이다.

### Goal

 우리는 처음에 video 부분과 audio 부분을 따로 처리한다. video의 경우에는 ViT(Vision Transformer) 모델을 이용하여 highlight와 non-highlight 부분울 학습시킨다. 이후 학습에 이용되지 않은 영상들을 모델에 넣어 ViT 모델이 분류한 데이터를 얻는다. audio의 경우에는 CNN(Convolutional Neural Network) 모델을 이용하여 video와 유사하게 highlight와 non-highlight 부분을 학습 및 분류시켜 데이터를 얻는다. 이후 video 부분과 audio 부분에서 얻은 데이터를 합하여 >>어떤 모델인지 적어주세요<< 모델을 이용하여 최종적으로 highlight 장면을 분류한다. 

# II. Datasets

SoccerNet의 축구 경기 영상 데이터를 이용하였다. 

https://www.soccer-net.org/home

SoccerNet은 축구 비디오 분석을 위한 대규모 데이터셋으로, 다양한 연구와 산업 응용을 지원하기 위해 개발되었다. Python 라이브러리를 통하여 쉽게 접근할 수 있지만, 비밀번호를 요구하기 때문에 SoccerNet 사이트에 들어가서 직접 신청서를 작성해야 한다. 우리는 이곳에서 유럽 리그의 축구경기 영상과 주요 장면이 시간별로 라벨링 된 데이터를 얻을 수 있었다. 

SoccerNet 라이브러리 다운로드

```python

$ pip install SoccerNet

```

Python에서 SoccerNet 라이브러리에 접근하는 코드

```python

import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader

mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory=r"SoccerNet")

mySoccerNetDownloader.password = "이곳에 비밀번호를 입력"

```

축구 경기 영상 다운로드

```python
mySoccerNetDownloader.downloadGames(files=["1_224p.mkv", "2_224p.mkv"], split=["train", "valid", "test"])

```

시간별로 라벨링 된 데이터 다운로드

```python
mySoccerNetDownloader.downloadGames(files=["Labels-v2.json"], split=["train", "valid", "test"])

```




