from logging import raiseExceptions
from detector.yolov5.rw_infer import infer


"""
return yolov5 model
    call do_infer(img_place) to do inference

"""
def load_yolov5():
    detector = infer()
    return detector


"""
return a detector model
based on name

"""
def load_detector(name='yolov5'):
    if name == 'yolov5':
        model = load_yolov5()
    else:
        print("Detector model not found")
        exit(0)
    return model
