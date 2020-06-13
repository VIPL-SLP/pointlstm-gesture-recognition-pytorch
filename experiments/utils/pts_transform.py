# This file is based on Pointnet2_Pytorch repo.
# https://github.com/erikwijmans/Pointnet2_PyTorch.git

from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import pdb
import numpy as np


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, pts):
        for t in self.transforms:
            pts = t(pts)
        return pts


def angle_axis(angle, axis):
    # type: (float, np.ndarray) -> float
    r"""Returns a 4x4 rotation matrix that performs a rotation around axis by angle
    Parameters
    ----------
    angle : float
        Angle to rotate by
    axis: np.ndarray
        Axis to rotate about
    Returns
    -------
    torch.Tensor
        3x3 rotation matrix
    """
    axis = axis.numpy()
    angle = angle.numpy()
    u = axis / np.linalg.norm(axis)
    cosval, sinval = np.cos(angle), np.sin(angle)

    cross_prod_mat = np.array([[0.0, -u[2], u[1]],
                               [u[2], 0.0, -u[0]],
                               [-u[1], u[0], 0.0]])

    R = torch.from_numpy(
        cosval * np.eye(3)
        + sinval * cross_prod_mat
        + (1.0 - cosval) * np.outer(u, u)
    )
    return R.float()


class PointcloudScale(object):
    def __init__(self, lo=0.8, hi=1.25):
        self.lo, self.hi = lo, hi

    def __call__(self, points):
        scaler = torch.FloatTensor(1).uniform_(self.lo, self.hi)
        points[:, :3] *= scaler
        return points


class PointcloudRotate(object):
    def __init__(self, axis=np.array([0.0, 1.0, 0.0])):
        self.axis = axis

    def __call__(self, points):
        rotation_angle = np.random.uniform() * 2 * np.pi
        rotation_matrix = angle_axis(rotation_angle, self.axis)

        normals = points.size(1) > 3
        if not normals:
            return torch.matmul(points, rotation_matrix.t())
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:]
            points[:, 0:3] = torch.matmul(pc_xyz, rotation_matrix.t())
            points[:, 3:] = torch.matmul(pc_normals, rotation_matrix.t())

            return points


class PointcloudRotatePerturbation(object):
    def __init__(self, angle_sigma=0.06, angle_clip=0.18):
        self.angle_sigma, self.angle_clip = angle_sigma, angle_clip

    def _get_angles(self):
        angles = torch.clamp(
            self.angle_sigma * torch.randn(3), -self.angle_clip, self.angle_clip
        )
        return angles

    def __call__(self, points):
        angles = self._get_angles()
        Rx = angle_axis(angles[0], torch.FloatTensor([1.0, 0.0, 0.0]))
        Ry = angle_axis(angles[1], torch.FloatTensor([0.0, 1.0, 0.0]))
        Rz = angle_axis(angles[2], torch.FloatTensor([0.0, 0.0, 1.0]))
        rotation_matrix = torch.matmul(torch.matmul(Rz, Ry), Rx)
        points[:, :3] = torch.matmul(points[:, :3], rotation_matrix.t())
        return points


class PointcloudJitter(object):
    def __init__(self, std=0.01, clip=0.05):
        self.std, self.clip = std, clip

    def __call__(self, points):
        # points[:, :3] += torch.clamp(torch.randn(1) * self.std, -self.clip, self.clip)
        points[:, :3] += torch.clamp(torch.randn(points[:, :3].shape) * self.std, -self.clip, self.clip)
        return points


# class PointcloudTranslate(object):
#     def __init__(self, translate_range=0.1):
#         self.translate_range = translate_range
#
#     def __call__(self, points):
#         translation = np.random.uniform(-self.translate_range, self.translate_range)
#         points[:, :3] += translation
#         return points


class PointcloudToTensor(object):
    def __call__(self, points):
        return torch.from_numpy(points).float()


class PointcloudRandomInputDropout(object):
    def __init__(self, max_dropout_ratio=0.875):
        assert max_dropout_ratio >= 0 and max_dropout_ratio < 1
        self.max_dropout_ratio = max_dropout_ratio

    def __call__(self, points):
        dropout_ratio = torch.rand(1) * self.max_dropout_ratio  # 0~0.875
        drop_idx = np.where(torch.rand((points.shape[0])).numpy() <= dropout_ratio.numpy())[0]
        if len(drop_idx) > 0:
            points[drop_idx] = points[drop_idx - 1]  # set to the first point

        return points
