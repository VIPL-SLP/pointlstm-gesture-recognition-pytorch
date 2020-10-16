import re
import pdb
import sys
import numpy as np
from utils import *
import torch.utils.data as data
from utils.pts_transform import *

sys.path.append("..")


class NvidiaLoader(data.Dataset):
    def __init__(self, framerate, valid_subject=None, phase="train", datatype="depth", inputs_type="pts"):
        self.phase = phase
        self.datatype = datatype
        self.inputs_type = inputs_type
        self.framerate = framerate
        self.valid_subject = valid_subject
        self.inputs_list = self.get_inputs_list()
        self.r = re.compile('[ \t\n\r:]+')

        print(len(self.inputs_list))
        if phase == "train":
            self.transform = self.transform_init("train")
        elif phase in ["test", "valid"]:
            self.transform = self.transform_init("test")

    def __getitem__(self, index):
        label = int(self.r.split(self.inputs_list[index])[-2])
        input_data = np.load(f"../dataset/{self.r.split(self.inputs_list[index])[1][1:-4]}_pts.npy").astype(float)
        input_data = self.normalize(input_data, self.framerate)
        return input_data, label, self.inputs_list[index]

    def get_inputs_list(self):
        prefix = "../dataset/Nvidia/Processed"
        if self.phase == "train":
            if self.datatype == "depth":
                inputs_path = prefix + "/train_depth_list.txt"
            elif self.datatype == "rgb":
                inputs_path = prefix + "/train_color_list.txt"
            inputs_list = open(inputs_path).readlines()
            ret_line = []
            for line in inputs_list:
                if "subject" + str(self.valid_subject) + "_" in line:
                    continue
                ret_line.append(line)
        elif self.phase == "valid":
            if self.datatype == "depth":
                inputs_path = prefix + "/train_depth_list.txt"
            elif self.datatype == "rgb":
                inputs_path = prefix + "/train_color_list.txt"
            inputs_list = open(inputs_path).readlines()
            ret_line = []
            for line in inputs_list:
                if "subject" + str(self.valid_subject) + "_" in line:
                    ret_line.append(line)
        elif self.phase == "test":
            if self.datatype == "depth":
                inputs_path = prefix + "/test_depth_list.txt"
            elif self.datatype == "rgb":
                inputs_path = prefix + "/test_color_list.txt"
            ret_line = open(inputs_path).readlines()
        else:
            AssertionError("Phase error.")
        return ret_line

    def __len__(self):
        return len(self.inputs_list)

    def normalize(self, pts, fs):
        timestep, pts_size, channels = pts.shape
        pts = pts.reshape(-1, channels)
        pts[:, 0] = (pts[:, 0] - np.mean(pts[:, 0])) / 120
        pts[:, 1] = (pts[:, 1] - np.mean(pts[:, 1])) / 160
        pts[:, 3] = (pts[:, 3] - fs / 2) / fs * 2
        if (pts[:, 2].max() - pts[:, 2].min()) != 0:
            pts[:, 2] = (pts[:, 2] - np.mean(pts[:, 2])) / (pts[:, 2].max() - pts[:, 2].min()) * 2
        pts = self.transform(pts)
        pts = pts.reshape(timestep, pts_size, channels)
        return pts

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

    @staticmethod
    def key_frame_sampling(key_cnt, frame_size):
        factor = frame_size * 1.0 / key_cnt
        index = [int(j / factor) for j in range(frame_size)]
        return index


if __name__ == "__main__":
    feeder = BaseFeeder(framerate=80)
    nvidia = torch.utils.data.DataLoader(
        dataset=feeder,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )
    for batch in nvidia:
        print(batch[0].shape, batch[1])
        pdb.set_trace()
