import os
import torch
import cv2
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from data_preparation import spectrogram_transition, clip_create, extract_audio
from torch.utils.data import DataLoader
from data_preprocess import prepare_dataset_audio, collate_fn_audio, prepare_dataset_video, collate_fn_video
import torch.nn as nn
import torch.optim as optim
from transformers import ViTForImageClassification
from model import ConvStridePoolSubsampling, Seq2VecGRU_Audio, Seq2VecGRU_Video
from transformers import ViTForImageClassification

root_dir = r"C:\Users\ksost\soccer_env\base_data\england_epl"
base_video_path = r"C:\Users\ksost\soccer_env\cliped_data\video"
base_audio_path = r"C:\Users\ksost\soccer_env\cliped_data\audio"
base_spectrogram_path = r"C:\Users\ksost\soccer_env\cliped_data\spectrogram"


# 전체 영상 하이라이트/비하이라이트 부분별로 나눈 뒤 저장
# clip_create.create_save(root_dir)

# # 영상에서 오디오 추출
# extracted_video_path = r"C:\Users\ksost\soccer_env\cliped_data\video"
# extract_audio.extract_audio_from_video(extracted_video_path, base_audio_path)

# # 오디오에서 스펙토그램 추출
# sepctogram = spectrogram_transition.extract_spectrogram_from_audio(base_audio_path, base_spectrogram_path)

# # 추출된 스펙터그램 데이터셋 구성
# audio_path = r"C:\Users\ksost\soccer_env\cliped_data\audio\highlights"
# audio_dataset = prepare_dataset_audio(audio_path, n_mels=64, augment=False)
# audio_data_loader = DataLoader(audio_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn_audio) 


video_path = r"/content/drive/MyDrive/AIX_DL_highlight_detector/highlights"
video_dataset = prepare_dataset_video(video_path)
video_data_loader = DataLoader(video_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_video)

model_1 = ConvStridePoolSubsampling()
model_2 = Seq2VecGRU_Audio()
# model_3 = ViT('B_32_imagenet1k', pretrained=True)
# model_3 = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
model_3 = resnet50(weights=ResNet50_Weights.DEFAULT)
model_4 = Seq2VecGRU_Video()

device = torch.device("cuda")
model_3 = model_3.to(device)  # 모델을 GPU로 이동

i = 0
for data in video_data_loader:
    data = data.to(device)  # 데이터를 GPU로 이동
    print(i)
    cliped_data = model_3(data)
    print(i)
    print(cliped_data.shape)
    i += 1

# BATCH_SIZE = 64
# LEARNING_RATE = 0.001
# EPOCHS = 10
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# # Define training loop
# def train(model, train_loader, criterion, optimizer, device):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()

#         if batch_idx % 100 == 0:
#             print(f'Train Batch {batch_idx}: Loss = {loss.item():.4f}')

# # Define evaluation loop
# def evaluate(model, test_loader, criterion, device):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += criterion(output, target).item()
#             pred = output.argmax(dim=1, keepdim=True)
#             correct += pred.eq(target.view_as(pred)).sum().item()

#     test_loss /= len(test_loader.dataset)
#     accuracy = 100. * correct / len(test_loader.dataset)
#     print(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')

# # Main function
# def main():
#     # 모델 불러오기
#     ViT = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

#     # Model, loss, optimizer
#     model_1 = ViT().to(DEVICE)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

#     # Training and evaluation
#     for epoch in range(1, EPOCHS + 1):
#         print(f'Epoch {epoch}/{EPOCHS}')
#         train(model, train_loader, criterion, optimizer, DEVICE)
#         evaluate(model, test_loader, criterion, DEVICE)

# if __name__ == '__main__':
#     main()



# # 데이터, 레이블 합치기
# labels = [0, 1, ...]  # 각 오디오에 해당하는 레이블 (예: 0: 일반, 1: 하이라이트)

# # DataLoader 생성
# dataloader = list(zip(audio_paths, labels))

# # 모델 학습
# train_ast(dataloader, model, criterion, optimizer)