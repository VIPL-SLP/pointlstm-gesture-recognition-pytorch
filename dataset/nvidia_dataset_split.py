import re
import os
import pdb
import glob
import numpy as np
import matplotlib.pyplot as plt


def str_insert(source_str, insert_str, pos):
    return source_str[:pos] + insert_str + source_str[pos:]


def load_data_from_file(avi_path, sensor, start_frame, end_frame, image_width=320, image_height=240, show_video=False):
    chnum = 3 if sensor == "color" else 1
    try:
        import skvideo.io
        video_container = skvideo.io.vread(avi_path)[start_frame:end_frame]
        if sensor != "color":
            video_container = video_container[..., 0][..., None]
    except ModuleNotFoundError:
        import cv2
        video_container = np.zeros((80, image_height, image_width, chnum), dtype=np.uint8)
        frames_to_load = range(start_frame, end_frame)
        cap = cv2.VideoCapture(avi_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for indx, frameIndx in enumerate(frames_to_load):
            ret, frame = cap.read()
            if ret:
                if sensor != "color":
                    frame = frame[..., 0][..., None]
                else:
                    frame = cv2.resize(frame, (image_width, image_height))
                video_container[indx] = frame
            else:
                print("Could not load frame")
    return video_container


def sensor_prefix(ss, spLine):
    if ss == "color":
        return spLine[2].split(":")[1]
    else:
        return spLine[1].split(":")[1]


if __name__ == "__main__":
    # sensors = ["color", "depth"]
    sensors = ["depth"]
    prefix = "./Nvidia"
    dataset = dict()
    dataset['train'] = os.path.join(prefix, "nvgesture_train_correct_cvpr2016_v2.lst")
    dataset['test'] = os.path.join(prefix, "nvgesture_test_correct_cvpr2016_v2.lst")

    for sensor in sensors:
        for data_path in dataset:
            read_stream = open(dataset[data_path], "r")
            read_stream = read_stream.readlines()
            if not os.path.exists(prefix + "/Processed"):
                os.makedirs(prefix + "/Processed")
            write_stream = open(f"{prefix}/Processed/{data_path}_{sensor}_list.txt", "w")
            print(f"{data_path}: total {len(read_stream)} videos.")
            r = re.compile('[ \t\n\r]+')
            for idx, line in enumerate(read_stream):
                if idx % 10 == 0:
                    print(f"{idx}/{len(read_stream)} videos are processed. ")
                splitLine = r.split(line)
                video_path = os.path.join(prefix + splitLine[0].split(".")[1],
                                          sensor_prefix(sensor, splitLine) + ".avi")
                sframe = int(splitLine[2].split(":")[2])
                eframe = int(splitLine[2].split(":")[3])
                video = load_data_from_file(video_path, sensor, sframe, eframe, 320, 240, show_video=False)
                label = int(splitLine[-2].split(":")[-1]) - 1
                save_prefix = video_path.split("Video_data")[0] + "Processed/" + data_path + \
                              video_path.split("Video_data")[1]
                if not os.path.exists(save_prefix):
                    os.makedirs(save_prefix)
                save_path = f"{save_prefix}/{str(idx).zfill(4)}_{sensor}_label_{str(label).zfill(2)}.npy"
                write_stream.write(str(idx).zfill(4) + "\t" + save_path + "\t" + str(label).zfill(2) + "\n")
                np.save(save_path, video)
