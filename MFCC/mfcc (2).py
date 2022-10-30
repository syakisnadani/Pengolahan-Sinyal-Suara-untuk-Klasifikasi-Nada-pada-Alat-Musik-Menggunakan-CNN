import os
import librosa
import math
import json
import numpy as np

# constant
DATASET_PATH = r"G:\Semester 5\Pemrosesan Suara\Project\Coding MFCC\dataset"
JSON_PATH = r"G:\Semester 5\Pemrosesan Suara\Project\Coding MFCC\mfcc.json"
# SAMPLE_RATE=22050
DURATION = 4  # in seconds


# SAMPLE_PER_TRACK=SAMPLE_RATE*DURATION

def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=1024, hop_length=512, num_segments=5):
    # dictionary
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }

    # loops dataset
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if (dirpath is not dataset_path):
            semantic_label = dirpath.split("/")[0]
            data["mapping"].append(semantic_label)
            print("\nProcessing {}".format(semantic_label))

            for f in filenames:
                # load file
                print("\nLoad Data {}".format(dirpath + '/' + f))
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path)

                number_samples_per_segment = int(sr / num_segments)
                enmvps = math.ceil(number_samples_per_segment / hop_length)  # expected_num_mfcc_vectors_per_segment

                # try cut non information wave
                for x in range(len(signal)):
                    if np.abs(signal[x]) >= 0.5:
                        awal = x
                        break
                cutsignal = signal[awal:len(signal)]
                for x in range(len(signal)):
                    if np.abs(signal[x]) >= 0.5:
                        akhir = x
                cutsignal = signal[0:akhir]
                if len(cutsignal) > 0:
                    signal = cutsignal
                print("Signal lengh: {}".format(len(signal)))
                if len(signal) == 0:
                    continue

                # process segment extracting mfcc and store it
                for s in range(num_segments):
                    start_sample = number_samples_per_segment * s
                    finish_sample = start_sample + number_samples_per_segment

                    if len(signal[start_sample:finish_sample]) == 0:
                        continue

                    mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample], sr=sr, n_mfcc=n_mfcc, n_fft=n_fft)
                    mfcc = mfcc.T

                    # if for filter only expected length
                    if len(mfcc) == enmvps:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i - 1)
                        print("{}, segment:{}".format(file_path, s))
    print(data)
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == '__main__':
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=3)