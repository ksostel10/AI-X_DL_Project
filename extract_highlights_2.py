import os
import cv2
import io
import torchaudio
import ffmpeg
import torchaudio.transforms as T
from torchaudio.transforms import FrequencyMasking, TimeMasking,  MelSpectrogram, AmplitudeToDB
import torch.nn as nn
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from AudioVideoDataset import MinMaxNormalize
from model import HighlightsClassifier
import numpy as np

audio_path = r"C:\Users\ksost\soccer_env\test\1_224p.wav"


# video_path = r"C:\Users\ksost\soccer_env\test\1_224p.mkv"


# # 변환 정의
# video_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Resize((224, 224)),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])

# # 비디오 캡처 객체 생성
# video = cv2.VideoCapture(video_path)

# # 비디오의 FPS 가져오기
# fps = video.get(cv2.CAP_PROP_FPS)
# frame_interval = int(fps / 5)  # 초당 5프레임을 위해 프레임 간격 계산

# # 모든 프레임을 저장할 리스트
# frames = []

# # 프레임 번호 초기화
# frame_count = 0

# # 비디오가 끝날 때까지 반복
# while True:
#     ret, frame = video.read()
    
#     if not ret:
#         break  # 비디오 끝났으면 반복 종료

#     # 지정된 프레임 간격에 따라 프레임을 선택
#     if frame_count % frame_interval == 0:
#         # BGR에서 RGB로 변환
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
#         # ndarray에서 PIL Image로 변환
#         frame = Image.fromarray(frame)
        
#         # 변환 적용
#         transformed_frame = video_transform(frame)
        
#         # 리스트에 추가
#         frames.append(transformed_frame)
    
#     frame_count += 1

# # 비디오의 모든 프레임을 텐서로 변환
# video_tensor = torch.stack(frames)

# output_path = r"C:\Users\ksost\soccer_env\test\video_tensor.pt"
# torch.save(video_tensor, output_path)

# video.release()


# output_path = r"C:\Users\ksost\soccer_env\test\video_tensor.pt"

# video_tensor = torch.load(output_path)

# waveform, _ = torchaudio.load(audio_path)

# if waveform.size(0) != 1:
#     waveform = waveform.mean(dim=0, keepdim=True)

# mel_spectrogram = MelSpectrogram(
#     sample_rate=48000, n_fft=400, hop_length=160, n_mels=64
# )
# to_db = AmplitudeToDB()
# normalizer = MinMaxNormalize()

# mel_spec = mel_spectrogram(waveform)
# mel_spec = to_db(mel_spec)
# mel_spec = normalizer(mel_spec)
# mel_spec = mel_spec.permute(2, 0, 1)  # (time, freq, channel) -> (channel, freq, time)

# output_path = r"C:\Users\ksost\soccer_env\test\audio_tensor.pt"
# torch.save(mel_spec, output_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HighlightsClassifier().to(device)
model.eval()

video_tensor_path = r"C:\Users\ksost\soccer_env\test\video_tensor.pt"
video_tensor = torch.load(video_tensor_path)
audio_tensor_path = r"C:\Users\ksost\soccer_env\test\audio_tensor.pt"
audio_tensor = torch.load(audio_tensor_path)


start_frame = 0
outputs = []

while True:
    end_frame = start_frame + 75
    video = video_tensor[start_frame:end_frame]
    audio = audio_tensor[start_frame*60:end_frame*60]
    audio = audio.unsqueeze(0)
    video.to(device)
    audio.to(device)
    audio = audio.permute(0, 2, 3, 1)
    with torch.no_grad():
        output = model(audio, video)
    print(output)
    output = output.cpu().numpy()
    outputs.append(output)
    if end_frame >= 13400:
        break
    start_frame += 15


print(outputs)

