import os
import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights


class ConvStridePoolSubsampling(torch.nn.Module):
    def __init__(self, conv_channels=32):
        super(ConvStridePoolSubsampling, self).__init__()
        self.conv_block = torch.nn.Sequential(
            # Conv Layer 1
            torch.nn.Conv2d(1, conv_channels, kernel_size=3, stride=2, padding=1),  # (64, 5984) -> (32, 2992)
            torch.nn.BatchNorm2d(conv_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # (32, 2992) -> (16, 1496)

            # Conv Layer 2x
            torch.nn.Conv2d(conv_channels, conv_channels * 2, kernel_size=3, stride=2, padding=1),  # (16, 1496) -> (8, 748)
            torch.nn.BatchNorm2d(conv_channels * 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # (8, 748) -> (4, 374)

            # Conv Layer 3
            torch.nn.Conv2d(conv_channels * 2, conv_channels * 4, kernel_size=3, stride=2, padding=1),  # (4, 374) -> (2, 187)
            torch.nn.BatchNorm2d(conv_channels * 4),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # (2, 187) -> (1, 187)

            # Conv Layer 4
            torch.nn.Conv2d(conv_channels * 4, 256, kernel_size=3, stride=(2, 2), padding=1),  # (1, 187) -> (1, 94)
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv_block(x)
        batch_size, channels, features, time = x.size()
        x = x.view(batch_size, time, -1)
        return x

class Seq2VecGRU_Audio(torch.nn.Module):
    def __init__(self, input_size=256, hidden_size=256, output_size=256, num_layers=1, bidirectional=True):
        super(Seq2VecGRU_Audio, self).__init__()
        
        # GRU 계층
        self.gru = torch.nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Fully Connected Layer (출력 크기 지정)
        self.fc = torch.nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, seq_len, input_size)
        """
        # GRU를 통해 시퀀스 처리
        _, hidden = self.gru(x)  # rnn_out: (batch_size, seq_len, hidden_size * num_directions)

        # 마지막 히든 상태 추출
        # hidden: (num_layers * num_directions, batch_size, hidden_size)
        # 양방향 GRU인 경우 양방향 히든 상태를 결합
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=-1)  # (batch_size, hidden_size * 2)

        # Fully Connected Layer 통과
        output = self.fc(hidden)  # (batch_size, output_size)

        return output
    
class Seq2VecGRU_Video(torch.nn.Module):
    def __init__(self, input_size=1000, hidden_size=512, output_size=512, num_layers=1, bidirectional=True):
        super(Seq2VecGRU_Video, self).__init__()
        
        # GRU 계층
        self.gru = torch.nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Fully Connected Layer (출력 크기 지정)
        self.fc = torch.nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, seq_len, input_size)
        """
        # GRU를 통해 시퀀스 처리
        _, hidden = self.gru(x)  # rnn_out: (batch_size, seq_len, hidden_size * num_directions)

        # 마지막 히든 상태 추출
        # hidden: (num_layers * num_directions, batch_size, hidden_size)
        # 양방향 GRU인 경우 양방향 히든 상태를 결합
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=-1)  # (batch_size, hidden_size * 2)

        # Fully Connected Layer 통과
        output = self.fc(hidden)  # (batch_size, output_size)

        return output

    
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
        self.softmax = torch.nn.Softmax(dim=1)

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
        x3 = self.softmax(x3)  # Softmax

        return x3