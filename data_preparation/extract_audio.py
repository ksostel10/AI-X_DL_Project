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

# video_path = r"C:\Users\ksost\soccer_env\test\real_test\1_224p.mkv"
# audio_path = r"C:\Users\ksost\soccer_env\test\real_test"
# def extract_audio_from_video(video_path, audio_path):
#     audio_path = os.path.join(audio_path, "1_224p.wav")
#     extract(video_path, audio_path)

# extract_audio_from_video(video_path, audio_path)