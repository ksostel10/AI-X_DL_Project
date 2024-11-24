import torchaudio
import torchaudio.transforms as T
import os
import matplotlib.pyplot as plt

def audio_to_spectrogram(audio_path, sample_rate=48000, n_fft=400, win_length=None, hop_length=160, n_mels=64):
    """
    오디오 파일을 멜 스펙트로그램으로 변환합니다.
    Args:
        audio_path (str): 오디오 파일 경로

    Returns:
        torch.Tensor: 멜 스펙트로그램 텐서
    """
    waveform, sr = torchaudio.load(audio_path)
    print(waveform.shape)

    # if waveform.size(0) == 2:
    #     waveform = waveform.mean(dim=0, keepdim=True)
    
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(waveform)
    
    spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels
    )(waveform)
    
    # 파워 스펙트로그램을 dB 단위로 변환
    spectrogram = torchaudio.transforms.AmplitudeToDB()(spectrogram)
    return spectrogram

def preprocess_audio(audio_path):
    """
    오디오를 멜 스펙트로그램으로 변환하고 AST 모델에 맞게 전처리합니다.
    Args:
        audio_path (str): 오디오 파일 경로

    Returns:
        dict: AST 모델 입력
    """
    #torchaudio.set_audio_backend("sox_io")
    
    spectrogram = audio_to_spectrogram(audio_path)
    spectrogram = spectrogram.squeeze(1).numpy()  # 모델 입력 형식에 맞게 변환
    return spectrogram

def save_spectrogram_image(spectrogram, save_path):
    """
    스펙트로그램을 이미지 파일로 저장합니다.
    
    Args:
        spectrogram (Tensor): 스펙트로그램 텐서
        save_path (str): 저장할 이미지 파일 경로
    """
    plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram, aspect='auto', origin='lower')
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel Spectrogram")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def extract_spectrogram_from_audio(base_audio_path, base_spectrogram_path):
    for target in os.listdir(base_audio_path):
        target_path = os.path.join(base_audio_path, target)
            
        for entry in os.listdir(target_path):
            audio_path = os.path.join(target_path, entry)
            spectrogram = preprocess_audio(audio_path)
            spectrogram_path = os.path.join(base_spectrogram_path, target, entry.split(".")[0] + ".png")
            save_spectrogram_image(spectrogram, spectrogram_path)