import cv2
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def key_frame_sampling(key_cnt, frame_size):
    factor = frame_size * 1.0 / key_cnt
    index = [int(j / factor) for j in range(frame_size)]
    return index


def uvd2xyz_sherc(pts, paras=(463.889, 463.889, 320.00000, 240.00000)):
    ret_pts = np.zeros_like(pts)
    ret_pts[:, 0] = (pts[:, 0] - paras[3]) * pts[:, 2] / paras[0]
    ret_pts[:, 1] = (pts[:, 1] - paras[2]) * pts[:, 2] / paras[1]
    ret_pts[:, 2] = pts[:, 2]
    ret_pts[:, 3] = pts[:, 3]
    return ret_pts


def uvd2xyz_nvidia(pts, paras=(224.50200, 230.49400, 160.00000, 120.00000)):
    ret_pts = np.zeros_like(pts)
    ret_pts[:, 0] = (pts[:, 0] - paras[3]) * (255 - pts[:, 2]) / paras[0]
    ret_pts[:, 1] = (pts[:, 1] - paras[2]) * (255 - pts[:, 2]) / paras[1]
    ret_pts[:, 2] = pts[:, 2]
    ret_pts[:, 3] = pts[:, 3]
    return ret_pts


def generate_pts_cloud_sequence(hand_crop_region, hand_regions, pts_size, time):
    pts_with_label = process_hand_crop(hand_crop_region, time)
    frame_pts = points_sampling(pts_with_label, pts_size)
    frame_pts[:, 0] += hand_regions[time][2]
    frame_pts[:, 1] += hand_regions[time][1]
    return frame_pts


def process_hand_crop(img, time):
    original_mask = np.zeros_like(img)
    original_mask[img > 0] = 1
    kernel = np.ones((7, 7), np.uint8)
    largest_mask = cv2.dilate(original_mask, kernel)
    largest_mask = cv2.erode(largest_mask, kernel)
    largest_mask = save_largest_label(largest_mask)
    original_mask *= largest_mask
    x, y = np.where(original_mask > 0)
    ret_info = np.zeros((len(x), 4))
    ret_info[:, 0] = x
    ret_info[:, 1] = y
    ret_info[:, 2] = img[x, y]
    ret_info[:, 3] = time
    return ret_info


def points_sampling(arr, cnt):
    if arr.shape[0] == 0:
        return np.zeros((cnt, arr.shape[1]))
    if arr.shape[0] < cnt:
        ind = np.arange(len(arr))
        ind = np.random.choice(ind, cnt, replace=True)
        arr_sampled = arr[ind]
    else:
        ind = np.arange(len(arr))
        ind = np.random.choice(ind, cnt, replace=False)
        arr_sampled = arr[ind]
    return arr_sampled


def generate_points(img, time):
    x, y = np.where(img != 0)
    if len(x) == 0:
        print("empty frame")
        return np.zeros((512, 4))
    hand_pts = np.zeros((len(x), 4))
    hand_pts[:, 0] = x
    hand_pts[:, 1] = y
    hand_pts[:, 2] = img[x, y]
    hand_pts[:, 3] = time
    return hand_pts


def save_largest_label(thresh):
    labels = measure.label(thresh, connectivity=1)
    regions = measure.regionprops(labels)
    if len(regions) > 1:
        areas = np.zeros(len(regions))
        for region_index, region in enumerate(regions):
            areas[region_index] = region.area
        sorted_order = np.argsort(areas, )
        thresh[np.where(labels != sorted_order[-1] + 1)] = 0
    return thresh


def topk_ind(mat1, mat2, k):
    dist = np.sum((mat1[:, None] - mat2[None]) ** 2, axis=-1) ** 0.5
    return np.argsort(dist, axis=1)[:, :k]


def insert(original, new, pos):
    '''Inserts new inside original at pos.'''
    return original[:pos] + new + original[pos:]


def show_video_point_clouds(video_point_cloud, continous=True, save_fig=None):
    video_length = len(video_point_cloud)
    x_min = y_min = 20000
    x_max = y_max = -20000
    for i in range(video_length):
        pts = video_point_cloud[i]
        y_min = min(pts[:, 0].min(), y_min)
        y_max = max(pts[:, 0].max(), y_max)
        x_min = min(pts[:, 1].min(), x_min)
        x_max = max(pts[:, 1].max(), x_max)
        length = max(y_max - y_min, x_max - x_min)
    if continous:
        plt.ion()
        fig = plt.figure()
        for i in range(video_length):
            show_frame_point_clouds(video_point_cloud[i], fig, (x_min, x_min + length),
                                    (y_min, y_min + length), show_img=False)
            plt.pause(0.5)
            plt.clf()
        plt.ioff()
        plt.close(fig)
    else:
        for i in range(video_length):
            if save_fig is not None:
                fig = plt.figure()
                show_frame_point_clouds(video_point_cloud[i], fig, (x_min, x_min + length),
                                        (y_min, y_min + length), show_img=False)
                plt.savefig(save_fig + str(i).zfill(3) + ".jpg")
                plt.close(fig)
            else:
                fig = plt.figure()
                show_frame_point_clouds(video_point_cloud[i], fig, (x_min, x_min + length), (y_min, y_min + length))


def show_frame_point_clouds(point_cloud, fig, x_minmax, y_minmax, show_img=True):
    ax = fig.add_subplot(111, projection='3d')
    ys = point_cloud[:, 0]
    xs = x_minmax[1] - point_cloud[:, 1] + x_minmax[0]
    zs = point_cloud[:, 2]
    ax.scatter(xs, ys, zs, c=zs, cmap=plt.get_cmap("jet"), alpha=1, s=5, vmin=100, vmax=500)
    ax.set_ylim(y_minmax[0], y_minmax[1])
    ax.set_xlim(x_minmax[0], x_minmax[1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=87, azim=90)
    if show_img:
        plt.show()
    return ax
