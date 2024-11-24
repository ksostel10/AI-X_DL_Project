import os
import json
import subprocess
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

def ffmpeg_extract_subclip_accurate(input_path, start_time, end_time, output_path):
    """
    정확히 동영상을 자르기 위해 FFmpeg의 -accurate_seek 옵션을 사용합니다.
    """
    command = [
        "ffmpeg",
        "-y",  # 출력 파일 덮어쓰기
        "-i", input_path,  # 입력 파일
        "-ss", str(start_time),  # 시작 시간
        "-to", str(end_time),  # 종료 시간
        "-c", "copy",  # 코덱 복사 (빠른 처리)
        "-avoid_negative_ts", "1",  # 타임스탬프 조정
        output_path
    ]
    subprocess.run(command, check=True)

def create_save(root_dir):
    # 각 경기 폴더를 순회
    for match_folder in os.listdir(root_dir):
        match_path = os.path.join(root_dir, match_folder)
        if os.path.isdir(match_path):
            # JSON 파일 경로 설정
            json_path = os.path.join(match_path, "Labels-v2.json")
            video1_path = os.path.join(match_path, "1_224p.mkv")  # 기본 비디오 파일 설정 (예: "1_224p.mp4")
            video2_path = os.path.join(match_path, "2_224p.mkv")

            # 동영상 파일이 있는지 확인
            if os.path.exists(video1_path) & os.path.exists(video2_path):
                # JSON 파일 로드
                with open(json_path, "r") as f:
                    label_data = json.load(f)

                for entry in label_data["annotations"]:
                    first_second_time = entry["gameTime"].split(" - ")[0]
                    time_str = entry["gameTime"].split(" - ")[1]
                    min, sec = map(int, time_str.split(":"))
                    event_time = min * 60 + sec

                    start_time = event_time - 10
                    end_time = event_time + 5
                    
                    if entry["label"] == "Goal" or entry["label"] == "Penalty" or entry["label"] == "Shots off target" or entry["label"] == "Shots on target":
                        label_dir = f"C:\\Users\\ksost\\soccer_env\\cliped_data\\video\\highlights"
                        clip_name = f"highlights_{len(os.listdir(label_dir)) + 1}.mkv"
                        output_path = os.path.join(label_dir, clip_name)

                        if first_second_time == "1":
                            ffmpeg_extract_subclip_accurate(video1_path, start_time, end_time, output_path)
                        else:
                            ffmpeg_extract_subclip_accurate(video2_path, start_time, end_time, output_path)
                    else:
                        label_dir = r"C:\Users\ksost\soccer_env\cliped_data\video\non-highlights"
                        clip_name = f"non-highlight_{len(os.listdir(label_dir)) + 1}.mkv"
                        output_path = os.path.join(label_dir, clip_name)

                        if first_second_time == "1":
                            ffmpeg_extract_subclip_accurate(video1_path, start_time, end_time, output_path)
                        else:
                            ffmpeg_extract_subclip_accurate(video2_path, start_time, end_time, output_path)