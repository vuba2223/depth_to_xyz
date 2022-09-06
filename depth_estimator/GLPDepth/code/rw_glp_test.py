
import os
import cv2
import numpy as np
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import depth_estimator.GLPDepth.code.utils.logging as logging
import depth_estimator.GLPDepth.code.utils.metrics as metrics
from depth_estimator.GLPDepth.code.models.model import GLPDepth
from depth_estimator.GLPDepth.code.dataset.base_dataset import get_dataset
from depth_estimator.GLPDepth.code.configs.test_options import TestOptions
from torchvision import transforms

torch.cuda.empty_cache()


metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log',
               'log10', 'silog']

class rw_glp_test():

    def __init__(self) -> None:
        opt = TestOptions()
        self.args = opt.initialize().parse_args()
        print("Params for GLP, can be modified at BaseOption class")
        print(self.args)
        if self.args.gpu_or_cpu == 'gpu':
            self.device = torch.device('cuda')
            cudnn.benchmark = True
            print("Using GPU for GLP Model")
        else:
            print("Using CPU for GLP Model")
            self.device = torch.device('cpu')

        if self.args.save_eval_pngs or self.args.save_visualize:
            result_path = os.path.join(self.args.result_dir, self.args.exp_name)
            logging.check_and_make_dirs(result_path)
            print("Saving result images in to %s" % result_path)
        
        if self.args.do_evaluate:
            self.result_metrics = {}
            for metric in metric_name:
                self.result_metrics[metric] = 0.0

        self.to_tensor = transforms.ToTensor()

        self.load_model()


    def load_model(self):
        print("\n1. Define Model")
        self.model = GLPDepth(max_depth=self.args.max_depth, is_train=False).to(self.device)
        model_weight = torch.load(self.args.ckpt_dir)
        if 'module' in next(iter(model_weight.items()))[0]:
            model_weight = OrderedDict((k[7:], v) for k, v in model_weight.items())
        self.model.load_state_dict(model_weight)
        self.model.eval()
        
    def load_images(self, data_path, data_type = 'imagepath'):
        print("\n2. Define Dataloader")
        if data_type == 'imagepath': # not for do_evaluate in case of imagepath
            dataset_kwargs = {'dataset_name': 'ImagePath', 'data_path': data_path}
        else:
            dataset_kwargs = {'data_path': data_path, 'dataset_name': self.args.dataset,
                            'is_train': False}

        test_dataset = get_dataset(**dataset_kwargs)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                pin_memory=True)
        return test_loader

    def estimate(self, test_loader):
        #ensure that memory will not be overloaded
        result_pairs = []
        print("\n3. Inference & Evaluate")
        for batch_idx, batch in enumerate(test_loader):
            input_RGB = batch['image'].to(self.device)
            print("Input_RGB shape: ", input_RGB.shape)
            filename = batch['filename']

            with torch.no_grad():
                pred = self.model(input_RGB)
            #original result *80 if it is trained by kitti dataset
            pred_d = pred['pred_d']

            pred_d_numpy = pred_d.squeeze().cpu().numpy()
            pred_d_numpy = (pred_d_numpy / pred_d_numpy.max()) * 255
            pred_d_numpy = pred_d_numpy.astype(np.uint8)
            pred_d_color = cv2.applyColorMap(pred_d_numpy, cv2.COLORMAP_RAINBOW)

            result_pairs.append((filename,pred_d, pred_d_color))
        return result_pairs
    
    def infer_with_path(self, image_path):
        result_pairs = []
        image = cv2.imread(image_path)  # [H x W x C] and C: BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # input size should be multiple of 32
        h, w, c = image.shape
        new_h, new_w = h // 32 * 32, w // 32 * 32
        image = cv2.resize(image, (new_w, new_h))
        # image = transforms.to_tensor(image)
        with torch.no_grad():
            pred = self.model(image)

        pred_d = pred['pred_d']
        pred_d_numpy = pred_d.squeeze().cpu().numpy()
        pred_d_numpy = (pred_d_numpy / pred_d_numpy.max()) * 255
        pred_d_numpy = pred_d_numpy.astype(np.uint8)
        pred_d_color = cv2.applyColorMap(pred_d_numpy, cv2.COLORMAP_RAINBOW)
        result_pairs.append((pred_d, pred_d_color))
        return result_pairs

    def infer(self, img):
        # result_pairs = []
        # image = cv2.imread(image_path)  # [H x W x C] and C: BGR
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # input size should be multiple of 32
        h, w, c = image.shape

        new_h, new_w = h // 32 * 32, w // 32 * 32
        image = cv2.resize(image, (new_w, new_h))
        
        image = self.to_tensor(image)
        image = image.unsqueeze(0)
        print("Input Image shape: ", image.shape)
        image = image.to(self.device)

        with torch.no_grad():
            pred = self.model(image)

        pred_d = pred['pred_d']
        # pred_d_numpy = pred_d.squeeze().cpu().numpy()
        # pred_d_numpy = (pred_d_numpy / pred_d_numpy.max()) * 255
        # pred_d_numpy = pred_d_numpy.astype(np.uint8)
        # pred_d_color = cv2.applyColorMap(pred_d_numpy, cv2.COLORMAP_RAINBOW)
        # result_pairs.append((pred_d, pred_d_color))

        # return pred_d
        return pred_d


    def do_infer(self, data_path, data_type = 'imagepath'):
        test_loader = self.load_images(data_path, data_type = 'imagepath')
        result = self.estimate(test_loader)
        return result