# 축구 하이라이트 영상 탐지, 추출, 생성기
24-2 한양대학교 AI-X 딥러닝 프로젝트

(영상) <br>
[https://youtu.be/7-xuGkrFdZg](https://youtu.be/sjSZGk7DpK8)

# Contents
Members

I. Proposal

II. Datasets

III. Methodology

IV. Model Architecture

V. Related Work

VI. Conclusion

<br>
<br>

# Members

김영일 / 경제금융학과 / il0906@naver.com

Task : 모델에 대한 설명글 작성 / 전체적인 블로그 글 작성 / 영상 촬영

정재희 / 경영학부 / ksostel10@naver.com

Task : 영상 원본에서 하이라이트 부분 추출해 학습데이터 구성하는 코드 작성 / 데이터 전처리(오디오 변환, 프레임추출, 데이터로더 구성 등) 코드 작성 / 영상 추출 코드 작성 / 학습 코드 돌리고 디버깅 / 코드 실행 순서 및 설명 관련 글 작성

양형석 / 전기공학부 / yhs30480@gmail.com

Task : 전체적인 모델 구조 설계 / 모델 코드 작성 / 데이터 전처리(패딩, 데이터 차원 조정 등 세부작업) 수행 / 영상 추출 코드 작성 / 학습 진행 코드 작성 / 전체적인 디버깅 작업 진행

<br>
<br>

# I. Proposal

### Motivation

 지금 우리 사회에서 사람들이 가장 많이 향유하고 있는 취미를 하나만 선정하자면 단연코 유튜브 시청일 것입니다. 많은 사람들이 영상으로 된 컨텐츠, 특히 유튜브 쇼츠의 등장으로 짧지만 큰 임팩트를 주는 영상을 선호하는 방향으로 흘러가고 있습니다. 이러한 사회적 상황에 맞게, 우리는 긴 영상에서 highlight 부분을 자동으로 추출해 줄 수 있는 딥러닝 모델을 구현하려 합니다. 

 그 중에서도 우리는 축구 경기 영상을 선택했습니다. 축구는 전반전과 후반전 각 45분에 추가시간까지 부여되기 때문에 총 영상의 길이가 100분 이상인 경우가 대부분입니다. 이 중에서 Goal(골), Penalty(페널티킥), Shots on target(유효슈팅), Shots off target(유효슈팅이 아닌 슈팅) 등 축구 경기 내에서 주목할 만한 부분들만 추출하여 영상으로 제작할 수 있다면 짧은 시간만으로 대부분의 축구 경기 내용을 파악하면서 더 재미있게 축구 영상을 시청할 수 있을 것라고 판단했습니다.

### Goal

 우리의 목표는 "축구 하이라이트 자동 추출 모델 만들기" 입니다. 이 목표를 위해서 진행되는 과정은 다음과 같습니다.
 
<br>
가장 먼저 경기당 100분정도 되는 축구영상을 highlight와 non-highlight로 분류합니다. 이때 하이라이트의 기준은 Goal(골), Peanlty(페널티킥), Shots on target(유효슈팅), Shots off target(유효슈팅이 아닌 슈팅)으로 지정하여 사건이 발생한 순간의 앞 10초, 뒤 5초를 포함하여 총 15초의 영샹을 하이라이트로 분류합니다. 그 외 나머지 부분은 non-highlight로 분류합니다.

<br>
<br>

하이라이트 부분에서는 관중, 해설자들의 소리가 커질 것으로 판단하여 일부 영상을 확인해 본 결과 실제로 오디오 크기와도 관계가 있음을 확인할 수 있었습니다. 따라서 video 부분뿐만 아니라 audio 부분도 포함하여 모델 학습을 진행하였습니다. audio의 경우에는 (Subsampling layer) CNN 모델을 이용하여 maxPooling 방식으로 길이를 줄이며 각 하이라이트의 중요한 특징을 추출하고, (Sequence to Vector)GRU 모델을 사용하여 시퀀스 데이터를 고정 길이 벡터로 변환하였습니다. video의 경우에는 초당 5프레임을 추출한 뒤 resNet 모델을 이용하여 각 하이라이트의 중요한 특징을 추출하고, (Sequence to Vector)GRU 모델을 사용하여 시퀀스 데이터를 고정 길이 벡터로 변환하였습니다. 즉, 저희는 전체 모델을 audio 관련 모델과 video관련 모델로 2개로 나누어 각각의 모델을 highlights, non-highlight 데이터로 학습시켰습니다. 
<br>

이때 audio 모델과 video 모델에서 나온 값이 fc layer를 통과하기 전에 결합한 뒤 fc layer에 통과시켜 audio, video 특징을 종합적으로 활용하여 highlight/non-highlights 이진 분류에 대한 확률을 도출할 수 있도록 하였습니다.

<br>

### Working Environment

Data preprocessing : Local(CPU : Intel(R) Core(TM) Ultra 5 125H, Memory : 16gb)

Model Training : Collab pro(GPU : A100 gpu)

<br>
<br>

# II. Datasets

## 1. 데이터 준비
<br>
https://www.soccer-net.org/home
<br>
<br>
데이터셋은 SoccerNet의 축구 경기 영상 데이터를 이용하였습니다. SoccerNet은 축구 비디오 분석을 위한 대규모 데이터셋으로, 다양한 연구와 산업 응용을 지원하기 위해 개발되었습니다. 아래의 코드를 실행하면 SoccerNet에서 제공하는 축구경기 영상과 라벨링 데이터를 얻을 수 있습니다. 코드는 "1_224p.mkv", "2_224p.mkv"의 전반 후반 영상 모두를 활용하는 방식으로 짜 놓았지만 실제로 저희는 로컬 저장장공간 부족으로 인해 "1_224p.mkv"만만 학습데이터 구성에 활용하였습니다.
<br>
<br>
SoccerNet 라이브러리 다운로드

```python
$ pip install SoccerNet
```

Python에서 SoccerNet 라이브러리에 접근하는 코드

```python
import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader

mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory=r"경로 입력")
mySoccerNetDownloader.password = "s0cc3rn3t"
mySoccerNetDownloader.downloadGames(files=["1_224p.mkv", "2_224p.mkv"], split=["train", "valid", "test"])
mySoccerNetDownloader.downloadGames(files=["Labels-v2.json"], split=["train", "valid", "test"])
```

<br>
<br>
<br>


![image](https://github.com/user-attachments/assets/ea0bcfab-3046-44b8-8ce9-326dc2b788df)

축구 경기영상은 다음과 같이 45분짜리 전반전 영상과 45분짜리 후반전 영상이 나뉘어서 제공됩

<br>
<br>

라벨링 데이터
<br>
![image](https://github.com/user-attachments/assets/695ef939-f2d3-4ebe-9053-ad61f64c6f72)
<br>
<br>
라벨링 데이터는 다음과 같이 json 형식으로 제공된다. label은 ball out of play, throw-in, foul, goal, penalty, shots on taget, shots off target 등 총 17개의 상황 중 하나로 지정돼있고 그 상황이 발생한 시점에 대해 "GameTime" : "전반/후반 - 시간" 의 시간 정보로 주어진다. 
<br>

## 2. 학습 데이터 추출
총 17개의 상황 중 Goal, Shots on target, Shots off target, Penalty 4개의 상황이 골, 슈팅 장면과 관련이 깊은 부분이라 판단하여 이를 하이라이트 장면의 판별 기준으로 삼고 전반/후반 경기 영상에서 이 부분들을 추출하여 학습 데이터셋을 구성하였습니다. 
<br>
### Video 데이터 추출
```python
# data_preparation/clip_create.py

def ffmpeg_extract_subclip_accurate(input_path, start_time, end_time, output_path):
    command = [
        "ffmpeg",
        "-y",  # 기존 파일 덮어쓰기
        "-i", input_path,  # 입력 파일
        "-ss", str(start_time),  # 시작 시간
        "-to", str(end_time),  # 종료 시간
        "-c:v", "libx264",  # 비디오 코덱: H.264 (효율적이고 호환성 높음)
        "-preset", "fast",  # 인코딩 속도와 품질의 균형 (fast 추천)
        "-crf", "23",  # 비디오 품질 설정 (낮을수록 고품질, 23은 적절한 기본값)
        "-c:a", "aac",  # 오디오 코덱: AAC (효율적이고 품질 유지)
        "-b:a", "128k",  # 오디오 비트레이트 (128 kbps는 일반적인 설정)
        "-strict", "experimental",  # AAC 관련 호환성
        output_path
    ]
    subprocess.run(command, check=True)

first_second_time = entry["gameTime"].split(" - ")[0]
time_str = entry["gameTime"].split(" - ")[1]
min, sec = map(int, time_str.split(":"))
event_time = min * 60 + sec

start_time = event_time - 10
end_time = event_time + 5
clip_key = (first_second_time, start_time, end_time)

if clip_key not in processed_clips:
    processed_clips.add(clip_key)  # 중복 방지

    if entry["label"] in ["Goal", "Penalty", "Shots off target", "Shots on target"]:
        label_dir = r"저장 폴더명"
        clip_name = f"highlights_{len(os.listdir(label_dir)) + 1}.mkv"
        output_path = os.path.join(label_dir, clip_name)

        if not os.path.exists(output_path):  # 중복 파일 방지
            if first_second_time == "1":
                ffmpeg_extract_subclip_accurate(video1_path, start_time, end_time, output_path)
            else:
                ffmpeg_extract_subclip_accurate(video2_path, start_time, end_time, output_path)
    else:
         cnt += 1
         if cnt == 8:
             label_dir = r"C:\Users\ksost\soccer_env\cliped_data\video\non-highlights"
             clip_name = f"non-highlight_{len(os.listdir(label_dir)) + 1}.mkv"
             output_path = os.path.join(label_dir, clip_name)

             if not os.path.exists(output_path):  # 중복 파일 방지
                 cnt = 0
                 if first_second_time == "1":
                     ffmpeg_extract_subclip_accurate(video1_path, start_time, end_time, output_path)
                 else:
                     ffmpeg_extract_subclip_accurate(video2_path, start_time, end_time, output_path)
             else:
                 cnt -= 1

root_dir = r"영상 데이터 경로 선택"
create_save(root_dir)
```
코드 설명: 영상 추출의 핵심 코드는 다음과 같습니다. json 파일에서 특정 장면의 시간정보를 가져와 해당 시점 10초 전, 5초 뒤 총 15초를 학습데이터의 길이로 설정하고 저희가 지정한 하이라이트 레이블일 경우 "highlights_n.mkv", 저희가 지정한 레이블이 아닐 경우 "non-highlight_n.mkv"로 구분하여 저정할 수 있도록 하였습니다. 학습데이터의 길이를 15초로 지정한 이유는 딱 골/슈팅 장면만 판별하는 것이 아니라 골/유효슈팅으로 이어지는 빌드업 장면까지 포함하여 학습할 수 있도록 하여 좀 더 완전한 하이라이트 장면을 추출할 수 있도록 하기 위함입니다. 이렇게 highlights/non-highlights에 해당하는 시간을 지정한 뒤 subproccess 모듈을 활용해 영상의 추출 작업을 수행하였습니다. 
<br>
<br>
실행방식: 코드에 폴더를 생성하는 부분을 넣어놓지 않아서 우선 로컬 공간에 몇개의 폴더가 준비돼있어야합니다. 저희는 cliped_data 퐅더 -> video, audio 폴더 -> 각 폴더 안에 highlights, non-highlights 폴더를 만들어 놓은 다음 하이라이트 영상 추출 코드를 실행하였습니다. 실행은 터미널에 python clip_create.py를 입력하시면 됩니다. 
<br>
<br>
### Audio 데이터 추출
다음은 추출된 15초짜리 개별 하이라이트 영상에서 오디오를 추출하는 작업입니다. 이 작업은 ffmpeg 모듈을 활용해 video 데이터에서 audio 부분만 추출하는 방식으로 수행하였습니다. 경로설정에 유의해 python extract_audio.py로 해당 파일을 실행하시면 오디오를 추출할 수 있습니다.
```python
# data_preparation/extract_audio.py

import ffmpeg
import os

def extract(video_path, audio_path):
    ffmpeg.input(video_path, ss=0, accurate_seek=None).output(audio_path, vn=None, acodec="pcm_s16le", ar=44100, avoid_negative_ts="make_zero").run()

def extract_audio_from_video(extracted_video_path, base_audio_path):
    for target in os.listdir(extracted_video_path):
        target_path = os.path.join(extracted_video_path, target)

        for entry in os.listdir(target_path):
            video_path = os.path.join(target_path, entry)
            audio_path = os.path.join(base_audio_path, target, entry.split(".")[0] + ".wav")
            extract(video_path, audio_path)


base_audio_path = r"C:\Users\ksost\soccer_env\cliped_data\audio"
extracted_video_path = r"C:\Users\ksost\soccer_env\cliped_data\video"
extract_audio_from_video(extracted_video_path, base_audio_path)
```

<br>

## 3. 데이터 전처리
```python
# AudioVideoDataset.py

class AudioVideoDataset(Dataset):
    def __init__(self, audio_dir, video_dir, n_mels=64, num_frames=5):
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.n_mels = n_mels
        self.num_frames = num_frames

        # 공통된 파일 이름을 찾기
        self.file_names = [
            f.split(".")[0]
            for f in os.listdir(audio_dir)
            if f.endswith(".wav") and os.path.exists(os.path.join(video_dir, f.split(".")[0] + ".mkv"))
        ]

        # 레이블 생성: 파일 이름이 "highlights"로 시작하면 1, 아니면 0
        self.labels = [
            1 if f.startswith("highlights") else 0 for f in self.file_names
        ]

        # 오디오 전처리
        self.mel_spectrogram = MelSpectrogram(
            sample_rate=48000, n_fft=400, hop_length=160, n_mels=n_mels
        )
        self.to_db = AmplitudeToDB()
        self.normalizer = MinMaxNormalize()
        self.spec_augment = SpecAugment() 

        # 비디오 전처리
        self.video_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=10),
            transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]

        # 레이블 가져오기
        label = self.labels[idx]

        # 오디오 데이터 처리
        audio_path = os.path.join(self.audio_dir, file_name + ".wav")
        waveform, _ = torchaudio.load(audio_path)

        if waveform.size(0) != 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        mel_spec = self.mel_spectrogram(waveform)
        mel_spec = self.to_db(mel_spec)
        mel_spec = self.normalizer(mel_spec)
        mel_spec = self.spec_augment(mel_spec)

        mel_spec = mel_spec.permute(2, 0, 1)  # (time, freq, channel) -> (channel, freq, time)

        # 비디오 데이터 처리
        video_path = os.path.join(self.video_dir, file_name + ".mkv")
        video_tensor = self._load_video_frames(video_path)

        return mel_spec, video_tensor, label

    def _load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % self.num_frames == 0:  # 지정된 프레임 간격으로 추출
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self.video_transform(frame)
                frames.append(frame)
            frame_count += 1
            if len(frames) == 75:  # 최대 75개의 프레임만 추출
                break

        cap.release()

        if len(frames) == 0:  # 비디오가 비어있으면 예외 처리
            raise ValueError(f"No frames extracted from video: {video_path}")

        return torch.stack(frames)

def collate_fn(batch):
  audio_batch, video_batch, label_batch = zip(*batch)

  # Audio 패딩
  audio_batch = pad_sequence(audio_batch, batch_first=True, padding_value=0)
  audio_batch = audio_batch.permute(0, 2, 3, 1)  # (batch, freq, time, channel)

  # Video 패딩
  video_batch = pad_sequence(video_batch, batch_first=True, padding_value=0)
  batch_size, sequence, channel, height, width = video_batch.shape
  video_batch = video_batch.view(batch_size * sequence, channel, height, width)

  # Labels
  label_batch = torch.tensor(label_batch)

  return audio_batch, video_batch, label_batch
```
<br>
다음은 데이터가 모델에 들어가도록 전처리 해주는 작업을 수행했습니다. AudioVideoDataset이라는 class 안에서 audio, video의 전처리 작업을 모두 수행하도록 한 뒤 audio 특징이 담긴 텐서인 mel_spec, video_tensor, label을 return 해주어 이들로 DataLoader를 구성할 수 있도록 해 주었습니다. 이때 label은 파일을 하나씩 불러올 때마다 파일 이름에 highlights가 있으면 1, 아니면 0로 지정하고 해당 파일의 비디오, 오디오 텐서를 반환할 때 같이 반환할 수 있도록 하였습니다. 
<br>
<br>
audio 데이터는 우선 stero type인 원본 audio를 양 쪽값의 평균치로 mono type으로 바꿔준뒤  멜 스펙토그램으로 변환하고 db(데시벨)형식으로 변환하여 소리의 크기에 따른 특징이 추출될 수 있도록 하였습니다.
<br>
<br>
video 데이터의 경우 우선 저희가 활용한 224p의 영상의 프레임은 25p/s입니다. 따라서 5프레임마다 하나씩 프레임을 추출하여 초당 5개의 프레임이 추출될 수 있도록 하였습니다. 그렇게 총 15초에 해당하는 75개의 프레임을 추출하였고 프레임마다 RandomResizedCrop, RandomRotation, GaussianBlur를 적용하여 data augmentation을 해 주었습니다. 
<br>
<br>
코드 실행: AudioVideoDataset.py에는 데이터를 전처리하는 클래스만 정의돼 있습니다. colab_main.py 파일에서 dataset을 로드할 때 해당 클래스를 불러와 오디오, 비디오가 있는 폴더의 경로를 넣어주면 바로 전처리가 수행됩니다. 아래 코드부분에서 확인하실 수 있습니다. 
<br>
<br>

```python
# colab_main.py

# 데이터셋 로드
dataset = AudioVideoDataset(audio_dir, video_dir)
val_size = int(len(dataset) * validation_split)
train_size = len(dataset) - val_size

# 데이터셋 분할
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
```
이는 colab_main.py에서 AudioVideoDataset을 불러와 train, valuation의 data_loader를 구성하는 과정입니다. 이때 패딩작업은 AudioVideoDatset.py 파일에서 collate_fn을 정의하여 audio, video가 각각 텐서의 차원에 맞게 패딩을 할 수 있도록 해 주었고 DataLoader의 파라미터에 정의된 collate_fn 함수를 가져와 패딩작업이 수행될 수 있도록 하였습니다.  
<br>
<br>
## 4. 모델 학습
```python
import os
import time  # 시간 측정을 위한 모듈
import torch
from torch.utils.data import DataLoader, random_split
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from AudioVideoDataset import AudioVideoDataset, collate_fn
from model import HighlightsClassifier

# 설정
audio_dir = "/content/drive/MyDrive/AIX_DL_highlight_detector/audio"  # 오디오 데이터 디렉토리 경로
video_dir = "/content/drive/MyDrive/AIX_DL_highlight_detector/video"  # 비디오 데이터 디렉토리 경로
batch_size = 4
learning_rate = 1e-4
num_epochs = 20
validation_split = 0.2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터셋 로드
dataset = AudioVideoDataset(audio_dir, video_dir)
val_size = int(len(dataset) * validation_split)
train_size = len(dataset) - val_size

# 데이터셋 분할
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

# 모델 초기화
model = HighlightsClassifier().to(device)

# 손실 함수 및 옵티마이저
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)

# ReduceLROnPlateau 스케줄러 추가
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# 학습 및 검증 루프
for epoch in range(num_epochs):
    start_time = time.time()  # 에포크 시작 시간 기록

    # 학습 단계
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for audio, video, labels in train_loader:
        audio = audio.to(device)
        video = video.to(device)
        labels = labels.to(device)

        # 옵티마이저 초기화
        optimizer.zero_grad()

        # 모델의 순전파
        outputs = model(audio, video)

        # 손실 계산
        loss = criterion(outputs, labels)
        train_loss += loss.item()

        # 역전파 및 매개변수 업데이트
        loss.backward()
        optimizer.step()

        # 정확도 계산
        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == labels).sum().item()
        train_total += labels.size(0)

    train_loss /= len(train_loader)
    train_accuracy = train_correct / train_total * 100

    # 검증 단계
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for audio, video, labels in val_loader:
            audio = audio.to(device)
            video = video.to(device)
            labels = labels.to(device)

            # 모델의 순전파
            outputs = model(audio, video)

            # 손실 계산
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # 정확도 계산
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= len(val_loader)
    val_accuracy = val_correct / val_total * 100

    # 스케줄러 업데이트 (검증 손실 기반)
    scheduler.step(val_loss)

    # 에포크 종료 시간 기록 및 경과 시간 계산
    end_time = time.time()
    epoch_time = end_time - start_time

    # 모델 저장
    model_save_path = os.path.join("/content/AIX_DL_highlight_detector/highlights_classifier/checkpoint", f"./highlights_classifier{epoch+1}.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # Epoch 결과 출력
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
    print(f"Epoch Time: {epoch_time:.2f} seconds")
```
colab_main.py는 코랩 환경에서 돌아갈 수 있도록 경로 설정을해 주었다. 각 epoch마다 학습과 검증단계를 거친 모델이 저장될 수 있도록 해 주었는데 자원 문제상 학습데이터를 많이 준비하지 못하였고 train, validation이 들쑥날쑥하며 증가하였을 뿐만 아니라(이 문제는 후술할 한계점/개선점 부분과 관련이 있음)  epoch 횟수도 20으로 짧아 20개의 모델을 전부 저장한 뒤 vaidation accuracy가 크면서 train accuracy가 그렇게 높지 않은 모델(과적합 방지)를 선택하여 실제 추론에 활용하였습니다. 저희는 14번째 epoch의 모델을 활용하였습니다. 아래 사진은 코랩의 A100을 통해 학습시킨 모델의 성능지표입니다.

<img src="https://github.com/user-attachments/assets/5dc990fa-7c2e-40d9-989f-ab45730325dc" width="700" height="700">
<img src="https://github.com/user-attachments/assets/d27ed445-c273-4c6f-a93d-0257cda8fe51" width="700" height="700">

<br>
<br>

## 5. 학습된 모델로 영상 추출

```python
import os
import cv2
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import torch
from torchvision import transforms
from PIL import Image
from AudioVideoDataset import MinMaxNormalize
from model import HighlightsClassifier
import os
import subprocess

audio_path = r"C:\Users\ksost\soccer_env\test\real_test\1_224p.wav"
video_path = r"C:\Users\ksost\soccer_env\test\real_test\1_224p.mkv"
audio_tensor_path = r"C:\Users\ksost\soccer_env\test\real_test\\audio_tensor.pt"
video_tensor_path = r"C:\Users\ksost\soccer_env\test\real_test\video_tensor.pt"
checkpoint_path = r"C:\Users\ksost\soccer_env\test\real_test\highlights_classifier14.pth"

# 비디오 45분풀영상 파일 텐서로 변환
video_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
video = cv2.VideoCapture(video_path)
fps = video.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps / 5)  # 초당 5프레임을 위해 프레임 간격 계산
frames = []
frame_count = 0
while True:
    ret, frame = video.read()
    if not ret:
        break  # 비디오 끝났으면 반복 종료
    # 지정된 프레임 간격에 따라 프레임을 선택
    if frame_count % frame_interval == 0:
        # BGR에서 RGB로 변환
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # ndarray에서 PIL Image로 변환
        frame = Image.fromarray(frame)
        # 변환 적용
        transformed_frame = video_transform(frame)
        # 리스트에 추가
        frames.append(transformed_frame)
    frame_count += 1
video_tensor = torch.stack(frames)

torch.save(video_tensor, video_tensor_path)
video.release()

# 오디오 45분짜리 파일 텐서로 변환
video_tensor = torch.load(video_tensor_path)
waveform, _ = torchaudio.load(audio_path)
if waveform.size(0) != 1:
    waveform = waveform.mean(dim=0, keepdim=True)
mel_spectrogram = MelSpectrogram(
    sample_rate=48000, n_fft=400, hop_length=160, n_mels=64
)
to_db = AmplitudeToDB()
normalizer = MinMaxNormalize()
mel_spec = mel_spectrogram(waveform)
mel_spec = to_db(mel_spec)
mel_spec = normalizer(mel_spec)
mel_spec = mel_spec.permute(2, 0, 1)  # (time, freq, channel) -> (channel, freq, time)
torch.save(mel_spec, audio_tensor_path)


def load_model_from_checkpoint(checkpoint_path, device):
    model = HighlightsClassifier().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 체크포인트에서 모델 상태 로드
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)  # 모델 상태만 저장된 경우 처리
    
    model.eval()
    print(f"Model loaded from checkpoint: {checkpoint_path}")
    return model

def extract_highlights_time(video_tensor, audio_tensor, model, device):
    outputs = []
    highlights_time = []
    start_frame = 0
    i = 0

    while True:
        end_frame = start_frame + 75

        if end_frame >= 12346:
            return highlights_time

        video = video_tensor[start_frame:end_frame]
        audio = audio_tensor[start_frame*55:end_frame*55]
        audio = audio.unsqueeze(0)
        video = video.to(device)
        audio = audio.to(device)
        audio = audio.permute(0, 2, 3, 1) # (batch, channel, freq, time)


        with torch.no_grad():
            output = model(audio, video)
        print(output)

        output = output.cpu().numpy()
        outputs.append(output)
        
        if output[0, 1] > 0.88:
            start_time = 3 * i
            end_time = start_time + 15

            # 연속으로 임계치를 넘었을 때 시작시간은 그대로, 끝나는 시간 가장 마지막껄로 바꿔줌
            if highlights_time and (start_time < highlights_time[-1][1]):
                highlights_time[-1] = (highlights_time[-1][0], end_time)
            else:
                highlights_time.append((start_time, end_time))
        print(highlights_time)
        
        start_frame += 15
        i += 1

def ffmpeg_extract_subclip_accurate(input_path, start_time, end_time, output_path):
    command = [
        "ffmpeg",
        "-y",  # 기존 파일 덮어쓰기
        "-i", input_path,  # 입력 파일
        "-ss", str(start_time),  # 시작 시간
        "-to", str(end_time),  # 종료 시간
        "-c:v", "libx264",  # 비디오 코덱: H.264
        "-preset", "fast",  # 인코딩 속도와 품질의 균형
        "-crf", "23",  # 비디오 품질
        "-c:a", "aac",  # 오디오 코덱
        "-b:a", "128k",  # 오디오 비트레이트
        output_path
    ]
    subprocess.run(command, check=True)

def create_save(video_path, highlights_time):
    # 동영상 파일이 있는지 확인
    if os.path.exists(video_path):
        for i, (start_time, end_time) in enumerate(highlights_time): 
            output_path = os.path.join(r"C:\Users\ksost\soccer_env\test\real_test", f"highlights_{i}.mkv")
            if not os.path.exists(output_path):
                ffmpeg_extract_subclip_accurate(video_path, start_time, end_time, output_path)


# 실행
video_tensor = torch.load(video_tensor_path)
audio_tensor = torch.load(audio_tensor_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 체크포인트에서 모델 불러오기
model = load_model_from_checkpoint(checkpoint_path, device)

# # 하이라이트 추출 및 동영상 저장
highlights_time = extract_highlights_time(video_tensor, audio_tensor, model, device)
create_save(video_path, highlights_time)
```
다음은 학습된 모델을 바탕으로 영상의 하이라이트 장면을 판별해 추출하는 코드입니다. 먼저 코랩에 저장해놨던 checkpoint 모델도 로컬로 직접 가져와 추론 모델을 준비시켜놓았습니다. 이후 학습에 이용돼지 않았던 다른 1_224p.mkv 파일을 준비하고 extract_aduio.py 파일을 활용해 오디오 파일을 생성해 주었습니다. 이후 extract_highlights.py 파일을 실행시키만 다음 부분들이 실행됩니다. <br> <br> 
전체 동영상의 텐서, 전체 오디오의 텐서를 추출해 audio_tensor.pt, video_tensor.pt 파일로 저장 <br><br>
해당 텐서를 모델에 통과시켜 하이라이트일 확률을 도출. 이때 duration이 15초일 때 프레임 구간을 계산하여 해당 부분을 추출하고 모델에 통과시킨 뒤 3초 간격으로 이동하면서 해당 작업을 반복해주었음 <br><br>
하이라이트로 판별되었으면 해당 텐서 부분의 start_time과 end_time을 리스트 변수에 저장. 이때 텐서는 프레임별로 저장돼 있기 때문에 224p영상이 초당 25프레임인 것을 활용해 프레임 위치로부터 시간 값 도출 가능 <br><br>
확보한 하이라이트 시간값들에 해당하는 영상을 추출 <br><br> 
이 작업들은 extract_highlights.py를 실행하면 자동으로 이루어집니다. 이 작업이 끝나면 하이라이트 영상들은 모두 추출되게 됩니다! 이어붙이는 작업에 해당하는 코드는 시간관계상 짜지 못하였고 개별적인 하이라이트 파일로만 존재합니다.



# III. Model Architecture
![image](https://github.com/user-attachments/assets/991be364-ada3-4eb6-8846-46096e22517e)

위의 그림은 저희가 만든 모델의 전체 구조입니다. 모델을 구성할 때 가장 시간이 많이 들었던 작업은 데이터가 여러 모델들을 통과할 수 있도록 차원을 조정해주는 작업이었습니다. 따라서 Input tensor의 변화를 자세히 살펴보겠습니다.

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
이를 적용한 코드는 다음과 같습니다.
```python
# model.py
class HighlightsClassifier(torch.nn.Module):
    def __init__(self):
        super(HighlightsClassifier, self).__init__()
        self.ConvStridePoolSubsampling = ConvStridePoolSubsampling()
        self.Seq2VecGRU_Video = Seq2VecGRU_Video()
        self.Seq2VecGRU_Audio = Seq2VecGRU_Audio()
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.fc1 = torch.nn.Linear(768, 768)
        self.fc2 = torch.nn.Linear(768, 2)
        self.audio_norm = torch.nn.LayerNorm(256)
        self.video_norm = torch.nn.LayerNorm(512)
        self.relu = torch.nn.ReLU()
        self.dropout_audio = torch.nn.Dropout(p=0.2)  # 오디오 경로 dropout
        self.dropout_video = torch.nn.Dropout(p=0.2)  # 비디오 경로 dropout
        self.dropout_fc = torch.nn.Dropout(p=0.3)     # Fully Connected Layer dropout

    def forward(self, x1, x2):
        # 오디오 경로 처리
        x1 = self.ConvStridePoolSubsampling(x1)
        x1 = self.Seq2VecGRU_Audio(x1)
        x1 = self.audio_norm(x1)  # LayerNorm 적용
        x1 = self.relu(x1)        # ReLU 적용
        x1 = self.dropout_audio(x1)  # Dropout 적용

        # 비디오 경로 처리
        num_frames, channels, height, width = x2.shape
        batch_size = num_frames // 75
        x2 = self.resnet(x2)
        x2 = x2.view(batch_size, num_frames // batch_size, -1)  # Reshape for GRU: (batch_size, num_frames, feature_size)
        x2 = self.Seq2VecGRU_Video(x2)
        x2 = self.video_norm(x2)  # LayerNorm 적용
        x2 = self.relu(x2)        # ReLU 적용
        x2 = self.dropout_video(x2)  # Dropout 적용

        # 두 경로 결합
        x3 = torch.cat((x1, x2), dim=1)

        # Fully Connected Layers
        x3 = self.fc1(x3)
        x3 = self.relu(x3)  # ReLU
        x3 = self.dropout_fc(x3)  # Dropout 적용
        x3 = self.fc2(x3)
        x3 = torch.softmax(x3, dim=1)

        return x3
```

# IV. Methodology

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

도출된 하이라이트 중 하나입니다. 추출된 모든 하이라이트는 위의 "detected_highlights" 파일에 올려두었습니다. 

https://github.com/user-attachments/assets/d0040fc1-71b2-4030-919f-84825a72dc2e



우리가 highlight로 지정했던 Shots off target을 잘 분류하는 모습이다.
<br>
<br>
<br>
<br>
최종적으로 우리는 SoccerNet에서 축구 경기 영상을 제공받아 highlight(Goal, Penatly, Shots on target, Shots off target)와 non-highlight로 분류하여 audio와 video로 나누어 학습시킨 후 Concatenate하여 highlight or non-highlight 이진분류를 수행하는 일련의 과정을 통해 축구 하이라이트 영상을 뽑아내겠다는 목표를 성공적으로 수행했습니다. 
<br>
<br>

###  한계점 및 개선점

<br>
- 너무 긴 학습영상
<br>
학습 데이터를 15초로 구성하였지만 이 중 실제 하이라이트 부분에 해당하는 슈팅장면, 골 장면은 2, 3초에 불과합니다. 나머지 12, 13초는 슈팅을 위한 빌드업 과정, 혹은 슈팅 이후 선수가 클로즈업되는 장면들 위주였습니다. 빌드업과정은 9초가량으로 총 영상 중 많은 비중을 차지하기 때문에 학습과정에서 더 치중되기 마련이고 선수가 클로즈업 되는 부분은 3~4초가량으로 짧더라도 이전 부분들(멀리서 영상이 찍힘)에 비해 확연하게 차이가 드러나기 때문에 더 치중되어 학습된 걸로 추정됩니다. 따라서 실제 판별장면에서는 골/슈팅장면보다 패스하는 빌드업 장면, 클로즈업되는 장면들이 많이 포함되는 결과가 나타났습니다.
<br>
<br>
이런 한계점은 학습데이터를 슈팅장면 전후 2~3초로 짧게 구성하여 학습시킨 뒤 슈팅장면이 판별되면 해당 시점에서 앞으로 8초, 뒤로 4~5초 가량을 덧붙여 추출하는 방식으로 개선할 수 있을 것으로 생각됩니다. 

<br>
<br>

### Source

- CNN <br>
https://blog.naver.com/rfs2006/223419122284 <br>
https://m.blog.naver.com/laonple/222344968031 <br>
<br>

- GRU <br>
https://wikidocs.net/22889 <br>
https://yjjo.tistory.com/18 <br>
<br>

- ResNet <br>
https://blog.naver.com/wooy0ng/222653802427 <br>
<br>

- ViT <br>
https://mishuni.tistory.com/137 <br>
https://daebaq27.tistory.com/108 <br>
https://gaussian37.github.io/dl-concept-vit/ <br>






