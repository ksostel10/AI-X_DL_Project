import ffmpeg
import os

def extract(video_path, audio_path):
    """
    비디오에서 오디오를 추출합니다.
    Args:
        video_path (str): 비디오 파일 경로
        audio_path (str): 추출된 오디오 파일 경로
    """
    stream = ffmpeg.input(video_path)
    stream = ffmpeg.output(stream, audio_path)
    ffmpeg.run(stream)

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