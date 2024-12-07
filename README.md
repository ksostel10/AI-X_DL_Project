# AI-X_DL_Project
24-2 한양대학교 AI-X 딥러닝 프로젝트

# Contents
Members

I. Proposal

II. Datasets

III. Methodology

IV. Model Architecture

V. Related Work

VI. Conclusion

# Members

김영일 / 경제금융학과 / il0906@naver.com

Task : 

정재희 /  / 

Task : 

양형석 / 전기공학부 / yhs30480@gmail.com

Task : 

# I. Proposal

### Motivation

 지금 우리 사회에서 사람들이 가장 많이 향유하고 있는 취미를 하나만 선정하자면 단연코 유튜브 시청일 것입니다. 많은 사람들이 영상으로 된 컨텐츠, 특히 유튜브 쇼츠의 등장으로 짧지만 큰 임팩트를 주는 영상을 선호하는 방향으로 흘러가고 있습니다. 이러한 사회적 상황에 맞게, 우리는 긴 영상에서 highlight 부분을 자동으로 추출해 줄 수 있는 딥러닝 모델을 구현하려 합니다. 

 그 중에서도 우리는 축구 경기 영상을 선택했습니다. 축구는 전반전과 후반전 각 45분에 추가시간까지 부여되기 때문에 총 영상의 길이가 100분 이상인 경우가 대부분입니다. 이 중에서 Goal(골), Penalty(반칙), Shots on target(유효슈팅), Shots off target(유효슈팅이 아닌 슈팅) 등 축구 경기 내에서 주목할 만한 부분들만 추출하여 영상으로 제작할 수 있다면 짧은 시간만으로 대부분의 축구 경기 내용을 파악하면서 더 재미있게 축구 영상을 시청할 수 있을 것라고 판단했습니다.

### Goal

우리의 목표는 >>   &nbsp;&nbsp;&nbsp;   << 입니다. 이 목표를 위해서 진행되는 과정은 다음과 같습니다. 우리는 처음에 audio 부분과 video 부분을 따로 처리합니다. 먼저 축구 경기 동영상에서 audio와 video를 분리하여 추출합니다. 이때, video는 초당 5프레임으로 설정하여 추출합니다. 이후 audio의 경우에는 (Subsampling layer) CNN 모델을 이용하여 maxPooling 방식으로 길이를 줄이며 각 하이라이트의 중요한 특징을 추출하고, (Sequence to Vector)GRU 모델을 사용하여 시퀀스 데이터를 고정 길이 벡터로 변환합니다. video의 경우에는 resNet 모델을 이용하여 각 하이라이트의 중요한 특징을 추출하고, (Sequence to Vector)GRU 모델을 사용하여 시퀀스 데이터를 고정 길이 벡터로 변환합니다. 마지막으로 audio 부분과 video 부분에서 얻은 데이터를 결합하여 하나의 Tensor를 생성하고, 이를 Fully Connected Layer 모델을 이용하여 최종적으로 highlight or non-highlight 이진 분류를 수행합니다. 

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

## Convolutional Neural Network(CNN)
CNN은 주로 공간적 데이터를 처리하기 위해 설계된 딥러닝 모델입니다. CNN은 Convolution(합성곱) 연산과 Pooling(서브샘플링)을 기반으로 데이터를 처리하며, 데이터의 공간적 구조를 효과적으로 학습합니다.

<br>

CNN의 구조는 아래와 같습니다.

<br>

1. 합성곱 계층(Convolutional Layer) <br>
이미지를 작은 패치로 분할하고, 각 패치에 가중치 행렬(필터)을 적용하여 특징을 추출합니다. 여러개의 필터를 사용하여 특징 추출 과정을 반복하여, 추출된 특징들은 특징맵(Feature Map)이라는 새로운 이미지 형태로 변환됩니다.

![image](https://github.com/user-attachments/assets/69b41e90-1712-45d1-be82-9bca62a58f2c)

<br>
<br>

2. 풀링 계층(Pooling Layer) <br>
특징맵의 크기를 줄여 계산량을 줄이고, 특징 정보의 중요성을 강조하는 역할을 합니다. 일반적으로 Max Pooling과 Average Pooling 두 가지 방식이 사용됩니다. 풀링 계층을 거치며 특징맵의 크기가 줄어들지만, 중요한 특징 정보는 유지됩니다.

![image](https://github.com/user-attachments/assets/49ee9e09-05cd-4630-ae71-f6a83da0034a)

<br>
<br>

3. 완전 연결 계층(Fully Connected Layer) <br>
이전 단계에서 추출된 특징맵을 벡터 형태로 변환하고, MLP를 사용하여 객체를 탐지하는 최종 출력을 생성합니다.

![image](https://github.com/user-attachments/assets/bcccdf70-ca9c-49a2-9d63-90c43df85ffb)

<br>

CNN의 중요한 기능 요소중 하나인 Subsampling layer에 대해서 알아보겠습니다.
<br>
Subsampling layer는 입력 데이터의 크기를 줄이는 역할을 합니다. 이는 데이터의 변화와 왜곡(위치이동, 회전, 부분적인 변화 등)에 강인한 인식 능력을 키워줍니다. 

데이터의 변형과 왜곡은 경우의 수가 셀 수 없을 정도로 많기 때문에, 이를 모두 준비하여 학습하는 것은 비효율적입니다. 이를 쉽게 해결하기 위해 subsampling을 활용합니다. subsampling을 통해 입력 데이터의 크기를 줄이면 비교적 강인한 특징만 남고, 자잘한 변화들은 사라지는 효과를 얻을 수 있습니다.

![image](https://github.com/user-attachments/assets/2d74f306-b90c-4333-b5fb-115043abe48c)

<br>
<br>

## Gated Recurrent Unit(GRU)

GRU는 RNN 기반 모델로, LSTM의 장기 의존성 문제에 대한 해결책을 유지하면서 은닉 상태를 업데이트하는 계산을 줄였습니다. 즉, LSTM과 비교했을 때 성능은 비슷하면서 구조를 간단화시켜 빠른 학습 속도를 가능하게 한 모델이라고 할 수 있습니다.

전체적인 GRU의 구조는 아래와 같습니다.

![image](https://github.com/user-attachments/assets/2155aef2-15a3-4599-86e2-90cb69d31f8e)
![image](https://github.com/user-attachments/assets/0b165e72-9e04-4797-9f97-138f9903405e)


<br>
<br>

각 단계를 순차적으로 알아보겠습니다.
<br>
1. Reset Gate <br>
과거의 정보를 적당히 리셋시키는게 목적으로 sigmoid 함수를 출력으로 이용해 (0, 1) 값을 이전 은닉층에 곱해줍니다.

![image](https://github.com/user-attachments/assets/dc11f22f-07f6-41fc-bdd5-9b1c7a1bbc2e)

<br>
<br>

2. Update Gate <br>
과거와 현재 정보의 최신화 비율을 결정합니다. sigmoid로 출력된 결과는 현시점의 정보의 양을 결정하고, (1-sigmoid) 값은 직전 시점의 은닉층의 정보에 곱해줍니다. 

![image](https://github.com/user-attachments/assets/3df0920c-b8b4-4428-8701-5de600d65982)

<br>
<br>

3. Candidate <br>
현 시점의 정보 후보군을 계산하는 단계입니다. 과거 은닉층의 정보를 그대로 이용하지 않고 리셋 게이트의 결과를 곱하여 이용해줍니다.

![image](https://github.com/user-attachments/assets/ea8d2978-b100-4931-a305-96f14f976b89)

<br>
<br>

4. 은닉층 계산 <br>
update gate 결과와 candidate 결과를 결합하여 현 시점의 은닉층을 계산하는 단계입니다. 앞에서 이야기했듯이 sigmoid 함수의 결과는 현시점의 정보 양을 결정하고, (1-sigmoid) 함수의 결과는 과거 시점의 정보 양을 결정합니다.

![image](https://github.com/user-attachments/assets/7583558b-7ad0-4d54-9379-3bf55acf84f0)

<br>
<br>

## Residual Network(ResNet)

ResNet은 Microsoft Research에서 2015년 발표한 딥러닝 모델로, 기존 CNN 모델에서 발생하는 기울기 소실(Vanishing Gradient) 문제를 해결하기 위해 제안되었습니다. 

ResNet의 핵심적인 아이디어는 Residual Connection(잔차연결)으로, 딥러닝에서 학습해야할 함수 H(x) 대신, 잔차(Residual)인 F(x)=H(x)−x를 학습하는 것입니다. 이 구조는 입력 x를 출력에 그대로 더하는 Skip Connection 구조로 구현됩니다.

이 구조를 그림으로 표현하면 아래와 같습니다.

![image](https://github.com/user-attachments/assets/2268e499-c8ab-45e0-b11b-59c1086b89e9)

Conv층을 통과한 F(X)와 Conv층을 통과하지 않은 x를 더하는 과정을 Residual Mapping 이라고 합니다. 위 Residual Block이 여러 개 쌓여서 나온 CNN 모델이 ResNet입니다.

모델명에 붙은 숫자는 층의 개수를 의미합니다. 우리가 이번 프로젝트에서 사용하는 모델은 ResNet-50으로, 50개의 층이 있다는 것을 의미합니다.


# IV. Model Architecture
![image](https://github.com/user-attachments/assets/991be364-ada3-4eb6-8846-46096e22517e)

위의 그림은 저희가 만든 모델의 전체 구조입니다. Input tensor의 변화를 자세히 살펴보겠습니다.
## Video part
그림에서 S는 Sequence, C는 Channel, H는 Height, W는 Width를 의미합니다.

Raw video의 (S, C, H, W)는 (**25*15, 3, 224, 398**)입니다.

저희는 영상 전처리 과정에서 Data Augmentation과 함께 Height와 Width를 (224, 224)로 Resizing 했습니다.
<br>또한 25fps를 5fps로 압축시키면서 Sequence의 길이를 5*15로 줄였습니다.

(**75, 3, 224, 224**)를 ResNet-50의 Input으로 사용했고, ImageNet-1K Pretrained Model을 사용한 만큼 output은 (**75, 1000**)이 됩니다.

이를 Bidirectional GRU에 통과시켰습니다.
<br>Hidden Size를 512로 사용하였고 Bidirectional이기 때문에 output은 (**512*2**)입니다.

이를 바로 Linear Layer에 통과시켜 (**512**)로 만듭니다.
<br>
<br>

## Audio part
Raw audio는 (Channel, Sequence, Amplitude)의 텐서입니다.

Channel의 경우 Stereo이므로 2이지만 양쪽 소리의 평균을 내어 1로 만들었습니다.
<br>그리고 Mel-Spectrogram 변환으로 STFT을 적용시켜 Sequence를 축소시키고 Amplitude를 Frequency 도메인으로 변환했습니다.
<br>그래서 전처리 과정을 거친 오디오 텐서는 (Channel, Sequence, Frequency)인 (**1, 4500, 64**)가 됩니다.

이를 stride가 2인 Convolution, MaxPooling Layer에 여러번 통과시켜 (Sequence/64, Feature), 즉 (**35, 2048**)을 output으로 갖습니다.

Video에서와 마찬가지로 Bidirectional GRU와 Linear Layer를 통과시켜 결국 (**256**)이 나오게 됩니다.
<br>
<br>

## Combining part
Video의 (**512**)와 Audio의 (**256**)를 Concatenate 하여 (**768**)을 얻습니다.
<br>얻은 벡터를 (768, 768), (768, 1)의 Linear Layer에 통과 시키면 (**1**)의 logit 스칼라가 됩니다.
<br>Logit이 Sigmoid function을 통과하고 결과적으로 저희가 원하는 Probability를 얻을 수 있게 됩니다.
<br>
<br>

# V. Related Work
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

위의 과정을 좀 더 상세히 기술하겠습니다. 먼저 이미지를 고정된 크기의 patch로 나누어야 합니다. 이 말은 기존의 (C, H, W) 크기의 이미지를 크기가 (P, P)인 patch N개로 자른다는 뜻입니다. 예를들어, C = 3, H = 32, W = 32 인 이미지가 있다면 이를 2X2 = 4개 patch의 16X16 이미지로 바꿀 수 있습니다. 즉, 4X(3X16X16) 형태가 되고, 이를 flatten 하게 1X768 크기의 vector로 변형하여 최종적으로 4X768의 matrix가 생성됩니다.

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

# VI. Conclusion

