import os
import cv2
import io
import torchaudio
import ffmpeg
import torchaudio.transforms as T
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import torch.nn as nn
import torch
from torchvision import transforms
from PIL import Image
from AudioVideoDataset import MinMaxNormalize
from model import HighlightsClassifier
import numpy as np
import os
import json
import subprocess
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

audio_path = r"C:\Users\ksost\soccer_env\test\real_test\1_224p.wav"
video_path = r"C:\Users\ksost\soccer_env\test\real_test\1_224p.mkv"
# video_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Resize((224, 224)),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])
# video = cv2.VideoCapture(video_path)
# fps = video.get(cv2.CAP_PROP_FPS)
# frame_interval = int(fps / 5)  # 초당 5프레임을 위해 프레임 간격 계산
# frames = []
# frame_count = 0
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
# video_tensor = torch.stack(frames)
# output_path = r"C:\Users\ksost\soccer_env\test\real_test\video_tensor.pt"
# torch.save(video_tensor, output_path)
# video.release()

# output_path = r"C:\Users\ksost\soccer_env\test\real_test\video_tensor.pt"
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
# output_path = r"C:\Users\ksost\soccer_env\test\real_test\\audio_tensor.pt"
# torch.save(mel_spec, output_path)




# 체크포인트 경로 설정
checkpoint_path = r"C:\Users\ksost\soccer_env\test\real_test\highlights_classifier14.pth"

def load_model_from_checkpoint(checkpoint_path, device):
    """
    체크포인트에서 모델을 불러옵니다.
    """
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
    """
    정확히 동영상을 자르기 위해 FFmpeg의 -accurate_seek 옵션을 사용합니다.
    """
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
video_path = r"C:\Users\ksost\soccer_env\test\real_test\1_224p.mkv"
video_tensor_path = r"C:\Users\ksost\soccer_env\test\real_test\video_tensor.pt"
video_tensor = torch.load(video_tensor_path)
audio_tensor_path = r"C:\Users\ksost\soccer_env\test\real_test\audio_tensor.pt"
audio_tensor = torch.load(audio_tensor_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 체크포인트에서 모델 불러오기
model = load_model_from_checkpoint(checkpoint_path, device)

# # 하이라이트 추출 및 동영상 저장
highlights_time = extract_highlights_time(video_tensor, audio_tensor, model, device)
create_save(video_path, highlights_time)