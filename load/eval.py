from load.cam_to_world import camxyz_to_worldxyz
from torchvision.transforms import Resize
import torch.nn.functional as F
from math import sqrt

def convert_boxes(boxes):
    return None



def metric(boxes, ground_truth, depth_map, center_size, camera_coordinate):
    """
    measurement of the distance of same location on grouth_truth and depth_map
    1. boxes in ground truth -> boxes in depth_map
    2. calculate the gnss of boxes of prediction
    3. calculate distances between boxes
    """

    gt_shape = ground_truth.shape[:1]
    d_shape = depth_map.shape[:1]


    if gt_shape != d_shape:
        boxes_in_d = convert_boxes(boxes)
    else:
        boxes_in_d = boxes

    for box in boxes_in_d:
        pass
    
    pass

def distance(pt1, pt2):
    dis = sqrt( (pt1[2]-pt2[2])**2 )
    # pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2 +

    return dis

def eval_an_image(converter, depth_estimator, rgb_image, boxes, roll, pitch, yaw, point_gts, cx, cy, cz,log = True, undistorted=True):
    """_summary_

    Args:
        converter (_type_): _description_
        depth_estimator (_type_): _description_
        rgb_image (_type_): _description_
        boxes (array of ints): x, y
        roll (_type_): _description_
        pitch (_type_): _description_
        yaw (_type_): _description_
        undistorted (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """


    if undistorted == False:
        return None
    else:
        depth_map = depth_estimator.infer(rgb_image)
        print("Depth map shape: ", depth_map.shape)
        #glp output has shape (1,1,h,w)
        # depth_map = depth_map.reshape(depth_map.shape[3], depth_map.shape[2]) #reshape to HxWx1
        # print("Depth map shape: ", depth_map.shape)
        # depth_map = Resize([2048, 2448])(depth_map)
        depth_map = F.interpolate(depth_map, (2048, 2448))
        print("Depth map shape after resize: ", depth_map.shape)
        depth_map = depth_map.reshape(2048, 2448) #reshape to HxWx1
        print("final depth map: ", depth_map.size())
        print("Depth = ", depth_map)
        print(depth_map)
        distances = []
        for id, box in enumerate(boxes):
            # x0 = int((box[2] - box[0])/2)
            # y0 = int((box[3] - box[1])/2)
            x0 = int(box[0])
            y0 = int(box[1])
            print("****************************")
            print("box: ", box)
            # print("x0: ", x0)
            # print("y0: ", y0)
            pw_gt = point_gts[id]
            pred_depth = depth_map[y0, x0].item()
            print("prediction depth: ", pred_depth)
            # x, y, depth, roll_deg, pitch_deg, yaw_deg, Cx, Cy, Cz
            pw = converter.pixel_to_world(x0, y0, pred_depth, roll, pitch, yaw, cx, cy, cz)
            dis = distance(pw_gt , pw)
            distances.append(dis)
            if log:
                print("gt: {}  // pred: {} // distance = {}".format(pw_gt, pw, dis))
        # avg_dis =  distances
        # if log:
        #     print

