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

김영일 / 경제금융학과 / il0906@naver.com

Task : 

정재희 /  / 

Task : 

양형석 /  /

Task : 

# I. Proposal

### Motivation

 지금 우리 사회에서 사람들이 가장 많이 향유하고 있는 취미를 하나만 선정하자면 단연코 유튜브 시청일 것입니다. 많은 사람들이 영상으로 된 컨텐츠, 특히 유튜브 쇼츠의 등장으로 짧지만 큰 임팩트를 주는 영상을 선호하는 방향으로 흘러가고 있습니다. 이러한 사회적 상황에 맞게, 우리는 긴 영상에서 highlight 부분을 자동으로 추출해 줄 수 있는 딥러닝 모델을 구현하려 합니다. 

 그 중에서도 우리는 축구 경기 영상을 선택했습니다. 축구는 전반전과 후반전 각 45분에 추가시간까지 부여되기 때문에 총 영상의 길이가 100분 이상인 경우가 대부분입니다. 이 중에서 Goal(골), Penalty(반칙), Shots on target(유효슈팅), Shots off target(유효슈팅이 아닌 슈팅) 등 축구 경기 내에서 주목할 만한 부분들만 추출하여 영상으로 제작할 수 있다면 짧은 시간만으로 대부분의 축구 경기 내용을 파악하면서 더 재미있게 축구 영상을 시청할 수 있을 것라고 판단했습니다.

### Goal

우리의 목표는 >>   &nbsp;&nbsp;&nbsp;   << 입니다. 이 목표를 위해서 진행되는 과정은 다음과 같습니다. 우리는 처음에 video 부분과 audio 부분을 따로 처리합니다. video의 경우에는 ViT(Vision Transformer) 모델을 이용하여 highlight와 non-highlight 부분울 학습시킵니다. 이후 학습에 이용되지 않은 영상들을 모델에 넣어 ViT 모델이 분류한 데이터를 얻습니다. audio의 경우에는 CNN(Convolutional Neural Network) 모델을 이용하여 video와 유사하게 highlight와 non-highlight 부분을 학습 및 분류시켜 데이터를 얻습니다. 이후 video 부분과 audio 부분에서 얻은 데이터를 합하여 >>   &nbsp;&nbsp;&nbsp;   << 모델을 이용하여 최종적으로 highlight 장면을 분류합니다. 

# II. Datasets

SoccerNet의 축구 경기 영상 데이터를 이용하였습니다. 

https://www.soccer-net.org/home

SoccerNet은 축구 비디오 분석을 위한 대규모 데이터셋으로, 다양한 연구와 산업 응용을 지원하기 위해 개발되었습니다. Python 라이브러리를 통하여 쉽게 접근할 수 있지만, 비밀번호를 요구하기 때문에 SoccerNet 사이트에 들어가서 직접 신청서를 작성해야 합니다. 우리는 이곳에서 유럽 리그의 축구경기 영상과 주요 장면이 시간별로 라벨링 된 데이터를 얻을 수 있었습니다다. 

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

<br>
<br>
<br>

위 코드를 실행하게 되면 SoccerNet 폴더가 생성되며, 그 안에 세부 폴더들을 들어가보면 

![image](https://github.com/user-attachments/assets/ea0bcfab-3046-44b8-8ce9-326dc2b788df)

이런 식으로 전반전/후반전으로 나뉜 축구 경기 영상과 라벨링된 데이터를 얻을 수 있습니다.

<br>
<br>

(라벨링 데이터 예시)
<br>
![image](https://github.com/user-attachments/assets/695ef939-f2d3-4ebe-9053-ad61f64c6f72)


# III. Methodology

## Video part
### ViT(Vision Transformer)
자연어 처리(NLP)에서 각광받는 Transformer 구조를 Language가 아닌 Computer Vision 영역에 적용한 모델입니다.

기존에 image sector에서는 attention 기법에 CNN모델을 함께 사용했던 것과 달리 attention 기법만을 사용하여(self-attention) 기존보다 높은 성능을 보이면서 처리 시간도 획기적으로 줄였다는 점에서 의의가 있다고 할 수 있습니다. 아래 사진을 보면서 ViT 모델의 sequence를 간단히 설명하겠습니다.

![image](https://github.com/user-attachments/assets/66dd2aba-9977-48b2-81fb-b8943d8a9a0a)

이미지를 고정된 크기의 patch로 나눠준 후, 각각의 이미지에 대하여 linear projection(선형 투영) 과정을 통해 embedding 작업을 합니다. 이후 원래 이미지의 정보를 추가해주기 위해 classification token을 더해주고, 이를 Transformer Encoder에 input값으로 넣습니다.

<br>
<br>
<br>
<br>
<br>
<br>

위의 과정을 좀 더 상세히 기술하겠습니다. 먼저 이미지를 고정된 크기의 patch로 나누어야 합니다. 이 말은 기존의 (C, H, W) 크기의 이미지를 크기가 (P, P)인 patch N개로 자른다는 뜻입니다. 예를들어, C = 3, H = 32, W = 32 인 이미지가 있다면 이를 2X2 = 4개 patch의 3X16X16 이미지로 바꿀 수 있습니다. 즉, 4X(3X16X16) 형태가 되고, 이를 flatten 하게 1X768 크기의 vector로 변형하여 최종적으로 4X768의 matrix가 생성됩니다.

![image](https://github.com/user-attachments/assets/05b1c7af-b211-4053-a713-10bde8816253)

<br>
<br>
<br>
<br>
<br>
<br>

다음으로, 위에서 만들어진 patch들을 patch embedding에 넣는 과정을 거칩니다. input vector들이 LayerNorm을 지나 linear layer에 들어갑니다. linear layer의 dimension을 3이라고 설정한다면 4X768의 데이터가 4X3으로 바뀌게 됩니다. 이후 다시 LayerNorm을 거쳐 최종 embedding vector가 생성됩니다.

![image](https://github.com/user-attachments/assets/76242f3d-d060-46fe-a5fb-b5a421a802f7)

<br>
<br>
<br>
<br>
<br>
<br>

이제 각 patch에 원래 이미지의 위치 정보를 추가해 줘야 하는데, 이를 Position Embedding이라고 합니다. Position Embedding을 진행하는 이유는 Transformer의 작동 방식과 관련이 있습니다. Transformer는 기존의 위치를 고려하지 않고 input data를 한번에 다 받기 때문에 위치 정보를 무시하게 됩니다. 이러한 상황에서 위치정보를 인위적으로 추가하면 기존보다 더 높은 성능을 이끌어낼 수 있습니다. 이러한 이유로 Position Embedding을 추가하여 진행합니다.

![image](https://github.com/user-attachments/assets/b1106861-153c-400d-93ad-2f060a146522)

<br>
<br>
<br>
<br>
<br>
<br>

마지막으로, 위에서 생성된 Final Embedding Vectors는 Dropout을 지나고 Transformer Block을 여러개 통과한 후에 MLP head를 지나 최종적으로 classification output이 나오게 됩니다.

![image](https://github.com/user-attachments/assets/799aa4a6-fae6-4d0d-9f42-d2ca789e935a)

<br>
<br>
<br>
<br>
<br>
<br>

이 과정을 애니메이션으로 표현하면 아래와 같습니다.

![vit](https://github.com/user-attachments/assets/6b9a036e-0a8d-42f8-9fea-b6d73fe97780)

https://github.com/lucidrains/vit-pytorch/blob/main/images/vit.gif

<br>
<br>

## Audio part


## Combining part


# IV. Evaluation & Analysis
## Video part

## Audio part

## Combining part


# V. Related Work

# VI. Conclusion

