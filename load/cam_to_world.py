from mimetypes import init
from typing_extensions import Self
import numpy as np
from numpy.linalg import inv
import math

PIXEL_SIZE = 3.45 * (10 ** -3)  # mm
FOCAL_LENGTH = 4.4  # mm
WIDTH = 2448  # pixels
HEIGHT = 2048  # pixels
Fx = 1239.359559  # pixels
Fy = 1239.398425  # pixels
Cx, Cy = 1203.818355, 1026.664654 #pixel
k1, k2, k3, k4 = -0.338192831, -0.018646323, 0.000379771, -0.001893033 #for undistorting
p1, p2 = 7.53878E-05, -6.42864E-05 #for undistorting

class camxyz_to_worldxyz():
    """
    U V W is the world coordinates
    X Y Z is the camera coordiates
    x y is the pixel coordiates

    assume that the depth value at the pixel position is Z
    1. from x y and Z find X Y Z by instrinsic
    2. from X Y Z and rotation matrix + translation to find U V W
    """
    def __init__(self, fx = Fx, fy = Fy, cx = Cx, cy = Cy, pixel_size = PIXEL_SIZE, image_width = WIDTH, image_height = HEIGHT):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.pixel_size = pixel_size
        self.width = image_width
        self.height = image_height
        
    def create_rotation_matrix(self, roll_deg, pitch_deg, yaw_deg): 
        """
        csv_file denotes yaw as heading
        parameter
        Roll - X
        Pitch - Y
        Yaw - Z
        R = Yaw * Pitch * Roll
        _________
        """
        roll = math.radians(roll_deg)
        pitch = math.radians(pitch_deg)
        yaw = math.radians(yaw_deg)
        r_x = np.array([
            [1, 0 , 0],
            [0, math.cos(roll), -math.sin(roll)],
            [0, math.sin(roll), math.cos(roll)]
        ])
        r_y = np.array([
            [math.cos(pitch), 0, math.sin(pitch)],
            [0, 1, 0],
            [-math.sin(pitch), 0, math.cos(pitch)]
        ])
        r_z = np.array([
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw), math.cos(yaw), 0],
            [0, 0 , 1]
        ])
        R = r_z.dot(r_y)
        R = R.dot(r_x)
        
        return R
    
    def cam_to_world(self, rotation_mat, Cx, Cy, Cz, X, Y, Z):
        """inspired by: https://www.fdxlabs.com/calculate-x-y-z-real-world-coordinates-from-a-single-camera-using-opencv/
        Calulate the Pw which is the world coordinates based on the camera coordites and the extrinsic matrix rotation and translation)

        Pc = R(Pw-C)
        [      [                 [              [ 
         X      r11 r12 r13 0     1  0  0 -Cx    U
         Y  =   r21 r22 r23 0  *  0  1  0 -Cy  * V
         Z      r31 r32 r33 0     0  0  1 -Cz    W
         1      0   0   0   1     0  0  0  1     1
          ]                  ]               ]    ]

        R*Temp the camera matrix and translation
        Pw = (R*Temp)^-1 * Pc

        Args:
            rotation_mat (3x3 mat): rotation matrix at that frame
            Cx (int): the x position of the cam origin in the world coordinates
            Cy (int): the y position of the cam origin in the world coordinates
            Cz (int): the z position of the cam origin in the world coordinates
            X (int): x position of the point in the camera coordiates
            Y (int): y position of the point in the camera coordiates
            Z (int): z position of the point in the camera coordiates

        Returns:
            1x3 array: Pw as world coordinate
        """

        Pc = np.array([
            X,
            Y,
            Z,
            1
        ])

        temp = np.array([
            [1,  0,  0, -Cx],
            [0,  1,  0, -Cy],
            [0,  0,  1, -Cz],
            [0,  0,  0,   1]
        ])

        print("rot mat: ", rotation_mat)

        rot_mat_4by4 = np.append(rotation_mat, [[0, 0, 0]], axis = 0)
        print("rot mat 4x4 : ", rot_mat_4by4)
        rot_mat_4by4 = np.append(rot_mat_4by4, [[0], [0], [0], [1]], axis = 1)
        print("rot mat 4x4 : ", rot_mat_4by4)
        Rxtemp = rot_mat_4by4.dot(temp)
        Pw = inv(Rxtemp).dot(Pc)
        print("original Pw: ", Pw)
        Pw = Pw / Pw[3]
        print("after Pw: ", Pw)
        return Pw[0:3]

    def pixel_to_cam(self, x, y, depth):
        """inspired by: OReilly Learning OpenCV Page:371
        given image coordinate (x, y)
        find cam coordinate (X, Y, Z)

        x_screen = fx*(X/Z) + Cx
        y_screen = fy*(Y/Z) + Cy

        cx, cy (pixel): the principal point of the image (not always the center of the image)
        fx, fy (pixel)

        Args:
            x (pixel): the coordinate (pixel) in the image plane (take s(0,0) as the origin of the image
            y (pixel): the coordinate (pixel) in the image plane (take s(0,0) as the origin of the image
            depth (meter): the distance from the camera to the object (in cam coordinate) which serves as Z in this case

        Returns:
            _type_: _description_
        """

        Z = depth
        X = Z*(x - self.cx) / self.fx
        Y = Z*(y - self.cy) / self.fy

        return [X, Y, Z]


        

    def pixel_to_world(self, x, y, depth, roll_deg, pitch_deg, yaw_deg, Cx, Cy, Cz):
        """ 1. pixel to cam
            2. cam to world
            image (x y) + depth -> camera coordinate (X Y Z) -> world coordinate (U V W)

        Args:
            x (pixel): the coordinate (pixel) in the image plane (take s(0,0) as the origin of the image
            y (pixel): the coordinate (pixel) in the image plane (take s(0,0) as the origin of the image
            depth (meter): the distance from the camera to the object (in cam coordinate) which serves as Z in this case
            roll_deg (degree): the degree to form the rotation matrix R
            pitch_deg (degree): the degree to form the rotation matrix R
            yaw_deg (degree): the degree to form the rotation matrix R
            Cx (int):  the world coordinate of the camera
            Cy (int):  the world coordinate of the camera
            Cz (int):  the world coordinate of the camera

        Returns:
            1x3 array: Pw the world coordinate of that image-pixel
        """
        rotation_mat = self.create_rotation_matrix(roll_deg, pitch_deg, yaw_deg)
        Pc = self.pixel_to_cam(x, y, depth)
        Pw = self.cam_to_world(rotation_mat, Cx, Cy, Cz, Pc[0], Pc[1], Pc[2])

        return Pw


# print("test camxyz_to_worldxyz")
# c = camxyz_to_worldxyz()
# rot_matrix = c.create_rotation_matrix(100, 100, 100)
