import os
import torchaudio

root_dir = r"C:\Users\ksost\soccer_env\cliped_data\audio"
def length_check(root_dir, list, i):
    # 각 경기 폴더를 순회
    for high_nonhigh in os.listdir(root_dir):
        audio_path = os.path.join(root_dir, high_nonhigh)
        for entry in os.listdir(audio_path):
            entry_path = os.path.join(audio_path, entry)
            waveform, _ = torchaudio.load(entry_path)
            list.append(waveform.shape[1])
            if waveform.shape[1] == 0:
                print(entry_path)
            # if i % 10 == 0:
            #     #print(min(list))
            i += 1

i = 0
list = []
length_check(root_dir, list, i)
print(min(list))