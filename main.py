from re import L
from load.load_detector import load_detector
from load.load_estimator import load_estimator
from load.image import load_image
from load.preprocessing import calib
import math
import cv2
import pandas as pd
from load.eval import eval_an_image
from load.cam_to_world import camxyz_to_worldxyz

def pred_image(img_place, ob_detector, dep_estimator):
    box_coordinates = ob_detector.do_infer(img_place)
    depth_list = dep_estimator.do_infer(img_place, data_type = 'imagepath')
    print("depth list len", len(depth_list))



def pred_an_image(img_path, is_calib, object_detector, depth_estimator):
    """predict an image location and depth map

    parameters
    ----------
    img: (H, W, 3)
        the original input from 

    calib: True
        remove distortion and turns to original size (2448, 2048)

    object_detector:
        yolov5
    
    depth_estimator:
        glp, adabin

    """
    if is_calib:
        img = calib(img_path)
    else:
        img = cv2.imread(img_path)
    
    boxes = object_detector.do_infer(img)
    depth_map = object_detector.do_infer(img)

    return boxes, depth_map




    
def main():
    df = pd.read_csv('verifying/ladybug_13451176_20200622_110245.csv')
    split_name = 'ladybug_13451176_20200622_110245_ColorProcessed_000336_Cam0_342_083-0353.png'.split('_')
    frame_name = '_'.join(split_name[2:4] + ['00' + split_name[5]])
    df.loc[df.FRAME == frame_name]

    cam_gps_location = df.loc[df.FRAME == frame_name][['Easting', 'Northing', 'H-Ell']].values
    roll = df.loc[df.FRAME == frame_name][['Roll']].values
    pitch = df.loc[df.FRAME == frame_name][['Pitch']].values
    yaw = df.loc[df.FRAME == frame_name]['Heading'].values

    print("Cam location: ", cam_gps_location[0])
    print("Roll: {}, pitch: {}, yaw: {}".format(roll, pitch, yaw))

    obj_coords = [[271893.125, 2768440, 90.05], [271888.218750, 2768442, 94.129997], [271895.406250, 2768437.75, 93.980003], [271893.125, 2768440, 92.629997], [271897.53125, 2768433.75, 88.27], [271893.125, 2768440, 89.42], [271887.375, 2768443, 94.129997]]

    coor_converter = camxyz_to_worldxyz()
    glp_model = load_estimator(name='glp')


    img = calib("/home/steve/thinktron/run_2/verifying/ladybug_13451176_20200622_110245_ColorProcessed_000336_Cam0_342_083-0353.png")
    print("Input image shape after Calib HxW: ", img.shape)
    
    anno = open("verifying/ladybug_13451176_20200622_110245_ColorProcessed_000336_Cam0_342_083-0353.txt")
    lines = anno.readlines()
    boxes = []

    used_H = 2048
    used_W = 2448

    for line in lines:
        segs = line.strip().split(" ")
        # print(line)
        # print(segs)
        x0 = int(float(segs[1])*used_W)
        y0 = int(float(segs[2])*used_H)
        x1 = x0 + int(float(segs[3])*used_W)
        y1 = y0 + int(float(segs[4])*used_H)
        box = [x0, y0, x1, y1]
        # print(box)
        boxes.append(box)
        # print("segs ", segs)

    eval_an_image(converter= coor_converter, depth_estimator=glp_model, rgb_image=img, boxes = boxes, 
                roll= roll, pitch=pitch, yaw=yaw, point_gts=obj_coords, cx=cam_gps_location[0][0], cy=cam_gps_location[0][1], cz=cam_gps_location[0][2])

    

main()





