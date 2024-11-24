import os
import cv2
import torchaudio
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
    

class SpecAugment(torch.nn.Module):     # 시간, 주파수 가리기
    def __init__(self, freq_mask_param=15, time_mask_param=35, num_masks=2):
        super(SpecAugment, self).__init__()
        self.freq_mask = FrequencyMasking(freq_mask_param)
        self.time_mask = TimeMasking(time_mask_param)

        self.num_masks = num_masks

    def forward(self, x): # 입력데이터는 (batch, freq, time) 형태를 가져야됨
        for _ in range(self.num_masks):
            x = self.freq_mask(x)
            x = self.time_mask(x)
        return x
    
# Custom Dataset
class SpecDataset(Dataset):
    def __init__(self, file_paths, n_mels=64, augment=True):
        self.file_paths = file_paths
        self.augment = augment

        # 전처리 모듈 초기화
        # self.resampler = ConvStridePoolSubsampling()
        self.mel_spectrogram = MelSpectrogram(
            sample_rate=48000, n_fft=400, hop_length=160, n_mels=n_mels
        )
        self.to_db = AmplitudeToDB()
        self.normalizer = MinMaxNormalize()
        self.spec_augment = SpecAugment() if augment else None
    
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # 1. 오디오 파일 로드
        file_path = self.file_paths[idx]
        waveform, _ = torchaudio.load(file_path)

        if waveform.size(0) != 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # 2. 데이터 증강 (선택적)
        if self.spec_augment:
            mel_spec = self.spec_augment(mel_spec)

        # 3. 멜 스펙트로그램 변환
        mel_spec = self.mel_spectrogram(waveform)
        mel_spec = self.to_db(mel_spec)  # 데시벨 변환

        # 4. 정규화
        mel_spec = self.normalizer(mel_spec)
        
        # 5. 차원 조정
        mel_spec = mel_spec.permute(2, 0, 1)

        return mel_spec

# 오디오 데이터셋 준비
def prepare_dataset_audio(data_dir, n_mels=64, augment=True):
    # 디렉토리 내 모든 .wav 파일 가져오기
    file_paths = [
        os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".wav")
    ]
    return SpecDataset(file_paths, n_mels=n_mels, augment=augment)

# 패딩을 추가하는 collate_fn 정의
def collate_fn_audio(batch):
    batch = pad_sequence(batch, batch_first=True, padding_value=0)
    batch = batch.permute(0, 2, 3, 1)
    # 배치 데이터를 패딩 처리 (길이가 서로 다른 시퀀스를 동일한 길이로 맞춤)
    return batch

class VideoDataset(Dataset):
    def __init__(self, file_path, num_frames=5):
        self.file_path = file_path
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=10),
            transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]) 
        self.num_frames = num_frames

    def __len__(self):
        return len(self.file_path)
    
    def __getitem__(self, idx):
        video_path = self.file_path[idx]

        # OpenCV를 사용해 영상 읽기
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0

        # 영상에서 지정된 프레임 수만큼 로드
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # 영상이 끝난 경우 중단
            if(frame_count%self.num_frames == 0):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR -> RGB
                frame = self.transform(frame)  # 프레임에 변환 적용
                frames.append(frame)
            frame_count += 1
        cap.release()

        # (num_frames, C, H, W) 형태로 변환
        video_tensor = torch.stack(frames)
        return video_tensor
    
# 영상 데이터셋 준비
def prepare_dataset_video(data_dir):
    file_paths = [
        os.path.join(data_dir, f) for f in os.listdir(data_dir)
    ]
    return VideoDataset(file_paths)

# 패딩을 추가하는 collate_fn 정의
def collate_fn_video(batch):

    # 배치 데이터를 패딩 처리 (길이가 서로 다른 시퀀스를 동일한 길이로 맞춤)
    batch = pad_sequence(batch, batch_first=True, padding_value=0)

    # (batch, sequence, channel, height, width) -> (batch * sequence, channel, height, width)
    batch_size, sequence, channel, height, width = batch.shape
    flattened_inputs = batch.view(batch_size * sequence, channel, height, width)
    
    return flattened_inputs