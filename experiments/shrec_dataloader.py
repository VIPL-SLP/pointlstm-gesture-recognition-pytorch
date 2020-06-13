import re
import pdb
import sys
import torch
import numpy as np
from utils import *
import torch.utils.data as data

sys.path.append("..")


def insert(original, new, pos):
    '''Inserts new inside original at pos.'''
    return original[:pos] + new + original[pos:]


class SHRECLoader(data.Dataset):
    def __init__(self, framerate, phase="train", datatype="depth", inputs_type="pts"):
        self.phase = phase
        self.datatype = datatype
        self.inputs_type = inputs_type
        self.framerate = framerate
        self.inputs_list = self.get_inputs_list()
        self.prefix = "../dataset/SHREC2017/gesture_{}/finger_{}/subject_{}/essai_{}"
        self.r = re.compile('[ \t\n\r:]+')
        print(len(self.inputs_list))
        if phase == "train":
            self.transform = self.transform_init("train")
        elif phase == "test":
            self.transform = self.transform_init("test")

    def __getitem__(self, index):
        splitLine = self.r.split(self.inputs_list[index])
        label28 = int(splitLine[-3]) - 1
        # label14 = int(splitLine[-4]) - 1
        input_data = np.load(
            insert(self.prefix.format(splitLine[0], splitLine[1], splitLine[2], splitLine[3]), "Processed_", 11)
            + "/pts_label.npy")[:, :, :7]
        input_data = input_data[self.key_frame_sampling(len(input_data), self.framerate)]
        for i in range(self.framerate):
            input_data[i, :, 3] = i
        input_data = np.dstack((input_data, np.zeros_like(input_data)))[:, :, :7]
        input_data = self.normalize(input_data, self.framerate)
        return input_data, label28, self.inputs_list[index]

    def get_inputs_list(self):
        prefix = "../dataset/SHREC2017"
        if self.phase == "train":
            inputs_path = prefix + "/train_gestures.txt"
        if self.phase == "test":
            inputs_path = prefix + "/test_gestures.txt"
        inputs_list = open(inputs_path).readlines()
        return inputs_list

    def __len__(self):
        return len(self.inputs_list)

    def normalize(self, pts, fs):
        timestep, pts_size, channels = pts.shape
        pts = pts.reshape(-1, channels)
        pts = pts.astype(float)
        pts[:, 0] = (pts[:, 0] - np.mean(pts[:, 0])) / 120
        pts[:, 1] = (pts[:, 1] - np.mean(pts[:, 1])) / 160
        pts[:, 3] = (pts[:, 3] - fs / 2) / fs * 2
        if (pts[:, 2].max() - pts[:, 2].min()) != 0:
            pts[:, 2] = (pts[:, 2] - np.mean(pts[:, 2])) / np.std(pts[:, 2])
        pts = self.transform(pts)
        pts = pts.reshape(timestep, pts_size, channels)
        return pts

    @staticmethod
    def key_frame_sampling(key_cnt, frame_size):
        factor = frame_size * 1.0 / key_cnt
        index = [int(j / factor) for j in range(frame_size)]
        return index

    @staticmethod
    def transform_init(phase):
        if phase == 'train':
            transform = Compose([
                PointcloudToTensor(),
                PointcloudScale(lo=0.9, hi=1.1),
                PointcloudRotatePerturbation(angle_sigma=0.06, angle_clip=0.18),
                # PointcloudJitter(std=0.01, clip=0.05),
                PointcloudRandomInputDropout(max_dropout_ratio=0.2),
            ])
        else:
            transform = Compose([
                PointcloudToTensor(),
            ])
        return transform


if __name__ == "__main__":
    dataloader = SHRECLoader(framerate=80)
    shrec = torch.utils.data.DataLoader(
        dataset=dataloader,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )
    for batch in shrec:
        print(batch[0].shape, batch[1])
        pdb.set_trace()
