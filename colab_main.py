import os
import time  # 시간 측정을 위한 모듈
import torch
from torch.utils.data import DataLoader, random_split
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