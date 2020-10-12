import numpy as np


def shift_img(img, h_size, ret_pts_origin_xy):
    """
    img: (H,W,3)
    h_size: useless size on the top.
    ret_pts_origin_xy: (N,2[W,H])
    """
    img = img[h_size:, :, :]
    ret_pts_origin_xy[:, 1] -= h_size
    return img, ret_pts_origin_xy

