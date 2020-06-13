import re
import os
import cv2
import pdb
import copy
import utils
import imageio
import numpy as np

if __name__ == "__main__":
    pts_size = 256
    r = re.compile('[ \t\n\r]+')
    dataset_prefix = "./SHREC2017"
    prefix = dataset_prefix + "/gesture_{}/finger_{}/subject_{}/essai_{}"
    train_list = open(dataset_prefix + "/train_gestures.txt").readlines()
    test_list = open(dataset_prefix + "/test_gestures.txt").readlines()
    input_list = train_list + test_list
    for idx, line in enumerate(input_list):
        # Loading dataset
        splitLine = r.split(line)
        dir_path = prefix.format(splitLine[0], splitLine[1], splitLine[2], splitLine[3])
        print(idx, len(input_list), dir_path)
        hand_regions = np.loadtxt(dir_path + '/general_informations.txt').astype(int)
        pts = np.zeros((32, pts_size, 8), dtype=int)
        ind = utils.key_frame_sampling(int(splitLine[-2]), 32)
        hand_regions = hand_regions[ind]
        depth_video = []
        for i, frame_id in enumerate(ind):
            # Reconstruct point cloud sequence from depth video
            depth_image = imageio.imread(dir_path + "/{}_depth.png".format(frame_id))
            depth_video.append(depth_image)
            hand_crop = depth_image[hand_regions[i][2]:hand_regions[i][2] + hand_regions[i][4],
                        hand_regions[i][1]:hand_regions[i][1] + hand_regions[i][3]]
            hand_crop = cv2.medianBlur(hand_crop, 3)
            pts[i, :, :4] = utils.generate_pts_cloud_sequence(hand_crop, hand_regions, pts_size, i)
            pts[i, :, 4:8] = utils.uvd2xyz_sherc(copy.deepcopy(pts[i, :, :4]))
        save_dir = utils.insert(dir_path, "Processed_", 2)
        try:
            os.makedirs(save_dir)
        except FileExistsError:
            pass
        np.save(save_dir + "/pts_label.npy", pts)
