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

class MinMaxNormalize(nn.Module):
    def __init__(self, min_val=0.0, max_val=1.0):
        super(MinMaxNormalize, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        x_min = x.min(dim=0, keepdim=True)[0]  # 최소값 (2D 기준)
        x_max = x.max(dim=0, keepdim=True)[0]  # 최대값 (2D 기준)

        # 정규화 공식: (x - x_min) / (x_max - x_min)
        x_normalized = (x - x_min) / (x_max - x_min + 1e-10)  # 1e-10은 분모 0 방지
        x_scaled = x_normalized * (self.max_val - self.min_val) + self.min_val

        return x_scaled
    

class AudioVideoDataset(Dataset):
    def __init__(self, audio_dir, video_dir, n_mels=64, num_frames=5):
        """
        Args:
            audio_dir (str): 오디오 파일 디렉토리 경로
            video_dir (str): 비디오 파일 디렉토리 경로
            n_mels (int): 멜 스펙트로그램의 멜 필터 개수
            augment (bool): 오디오 데이터 증강 여부
            num_frames (int): 비디오에서 추출할 프레임 간격
        """
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.n_mels = n_mels
        self.num_frames = num_frames

        # 오디오 전처리
        self.mel_spectrogram = MelSpectrogram(
            sample_rate=48000, n_fft=400, hop_length=160, n_mels=n_mels
        )
        self.to_db = AmplitudeToDB()
        self.normalizer = MinMaxNormalize()

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

        return mel_spec, video_tensor

    def _load_video_frames(self, video_path):
        """
        비디오 파일에서 프레임을 추출합니다.
        Args:
            video_path (str): 비디오 파일 경로
        Returns:
            torch.Tensor: (num_frames, channels, height, width) 형태의 텐서
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        cnt = 0
        fps = 25
        
        while cap.isOpened():
            start_frame = cnt
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % self.num_frames == 0:  # 지정된 프레임 간격으로 추출
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self.video_transform(frame)
                frames.append(frame)
            frame_count += 1
            if len(frames) == 75:  # 최대 75개의 프레임만 추출
                end_frame = cnt + frame_count
                start_time = start_frame // fps
                end_time = end_frame // fps
                
                # 오디오 추출
                # FFmpeg 명령 실행
                out, _ = (
                        ffmpeg
                        .input(self.video_dir, ss=start_time, to=end_time)
                        .output('pipe:', format='mp3', qscale=0, map='a')
                        .run(capture_stdout=True, capture_stderr=True)
                    )

                # 오디오 데이터를 메모리 내에서 다룰 수 있도록 BytesIO 객체로 변환
                audio_data = io.BytesIO(out)

                frame_count = 0
                


            cnt += 75

        cap.release()

        if len(frames) == 0:  # 비디오가 비어있으면 예외 처리
            raise ValueError(f"No frames extracted from video: {video_path}")

        return audio_data, torch.stack(frames)


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