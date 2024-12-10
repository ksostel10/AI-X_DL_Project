import os
import json
import subprocess

def ffmpeg_extract_subclip_accurate(input_path, start_time, end_time, output_path):
    command = [
        "ffmpeg",
        "-y",  # 기존 파일 덮어쓰기
        "-i", input_path,  # 입력 파일
        "-ss", str(start_time),  # 시작 시간
        "-to", str(end_time),  # 종료 시간
        "-c:v", "libx264",  # 비디오 코덱: H.264 (효율적이고 호환성 높음)
        "-preset", "fast",  # 인코딩 속도와 품질의 균형 (fast 추천)
        "-crf", "23",  # 비디오 품질 설정 (낮을수록 고품질, 23은 적절한 기본값)
        "-c:a", "aac",  # 오디오 코덱: AAC (효율적이고 품질 유지)
        "-b:a", "128k",  # 오디오 비트레이트 (128 kbps는 일반적인 설정)
        "-strict", "experimental",  # AAC 관련 호환성
        output_path
    ]
    subprocess.run(command, check=True)

def create_save(root_dir):
    processed_clips = set()

    # 각 경기 폴더를 순회
    for match_folder in os.listdir(root_dir):
        cnt = 0
        match_path = os.path.join(root_dir, match_folder)
        if os.path.isdir(match_path):
            # JSON 파일 경로 설정
            json_path = os.path.join(match_path, "Labels-v2.json")
            video1_path = os.path.join(match_path, "1_224p.mkv") # 기본 비디오 파일 설정 (예: "1_224p.mp4")
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
                    clip_key = (first_second_time, start_time, end_time)

                    if clip_key not in processed_clips:
                        processed_clips.add(clip_key)  # 중복 방지

                        if entry["label"] in ["Goal", "Penalty", "Shots off target", "Shots on target"]:
                            label_dir = r"C:\Users\ksost\soccer_env\cliped_data\video\highlights"
                            clip_name = f"highlights_{len(os.listdir(label_dir)) + 1}.mkv"
                            output_path = os.path.join(label_dir, clip_name)

                            if not os.path.exists(output_path):  # 중복 파일 방지
                                if first_second_time == "1":
                                    ffmpeg_extract_subclip_accurate(video1_path, start_time, end_time, output_path)
                                else:
                                    ffmpeg_extract_subclip_accurate(video2_path, start_time, end_time, output_path)

                        else:
                            cnt += 1
                            if cnt == 8:
                                label_dir = r"C:\Users\ksost\soccer_env\cliped_data\video\non-highlights"
                                clip_name = f"non-highlight_{len(os.listdir(label_dir)) + 1}.mkv"
                                output_path = os.path.join(label_dir, clip_name)

                                if not os.path.exists(output_path):  # 중복 파일 방지
                                    cnt = 0
                                    if first_second_time == "1":
                                        ffmpeg_extract_subclip_accurate(video1_path, start_time, end_time, output_path)
                                    else:
                                        ffmpeg_extract_subclip_accurate(video2_path, start_time, end_time, output_path)
                                else:
                                    cnt -= 1

root_dir = r"C:\Users\ksost\soccer_env\base_data\champs_and_epl"
create_save(root_dir)