import cv2
import numpy as np
KITTI_SIZE = (370,1224)

def calib(path):
    img = cv2.imread(path)[..., ::-1]
    # img = cv2.imread(path)

    img = cv2.resize(img, (2448, 2048))
    fx, fy = 1239.359559, 1239.398425
    Cx, Cy = 1203.818355, 1026.664654
    k1, k2, k3, k4 = -0.338192831, -0.018646323, 0.000379771, -0.001893033
    p1, p2 = 7.53878E-05, -6.42864E-05

    K = np.array([[fx, 0, Cx],
                [0, fy, Cy],
                [0, 0, 1]])
    D = np.array([k1, k2, k3, k4])

    # We need to assume D = [0, 0, 0, 0], instead of [k1, k2, k3, k4]
    mapx_inner, mapy_inner = cv2.fisheye.initUndistortRectifyMap(K, np.zeros(4), None, K,
                                                                (2448, 2048), cv2.CV_16SC2)

    undistort_img = cv2.remap(img, mapx_inner, mapy_inner, interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT)

    undistort_img = cv2.cvtColor(undistort_img, cv2.COLOR_BGR2RGB)

    return undistort_img
