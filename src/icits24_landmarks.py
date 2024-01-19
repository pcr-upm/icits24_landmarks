#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Juan Castrillo'
__email__ = 'juan.castrillo@alumnos.upm.es'

import os
import torch
import numpy as np
from enum import Enum
from images_framework.src.alignment import Alignment
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)


class Backbone(Enum):
    EDGENEXT = 'EdgeNeXt'
    MOBILENETV2 = 'MobileNetV2'
    CONVNEXT = 'ConvNeXt'


class ICITS24Landmarks(Alignment):
    """
    Object alignment using ICITS 2024 algorithm
    """
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.model = None
        self.device = None
        self.backbone = None
        self.width = 128
        self.height = 128

    def parse_options(self, params):
        unknown = super().parse_options(params)
        import argparse
        parser = argparse.ArgumentParser(prog='Ocr20Segmentation', add_help=False)
        parser.add_argument('--gpu', dest='gpu', type=int, action='append',
                            help='GPU ID (negative value indicates CPU).')
        parser.add_argument('--backbone', dest='backbone', required=True, choices=[x.value for x in Backbone],
                            help='Select backbone model.')
        args, unknown = parser.parse_known_args(unknown)
        print(parser.format_usage())
        mode_gpu = torch.cuda.is_available() and -1 not in args.gpu
        self.device = torch.device('cuda:{}'.format(args.gpu[0]) if mode_gpu else 'cpu')
        self.backbone = args.backbone

    def train(self, anns_train, anns_valid):
        print('Training ...')

    def load(self, mode):
        from images_framework.src.constants import Modes
        # Set up a neural network to train
        print('Load model')
        if self.backbone == 'EdgeNeXt':
            from .models.MobileViTs.EdgeNeXt.edgenext_l import edgenext_l_base
            self.model = edgenext_l_base(98).to(self.device)
        elif self.backbone == 'MobileNetV2':
            from .models.MobileNets.mobilenetv2 import mobilenetv2
            self.model = mobilenetv2(98).to(self.device)
        elif self.backbone == 'ConvNeXt':
            from .models.ConvNeXt.convnext import convnext_atomic
            self.model = convnext_atomic().to(self.device)
        if mode is Modes.TEST:
            model_file = self.path + 'data/' + self.database + '/' + self.backbone + '.pt'
            print('Loading model from {}'.format(model_file))
            self.model.load_state_dict(torch.load(model_file))
            self.model.eval()

    def process(self, ann, pred):
        import cv2
        # import itertools
        from images_framework.src.datasets import Database
        from images_framework.src.annotations import GenericLandmark
        from images_framework.alignment.landmarks import lps
        datasets = [subclass().get_names() for subclass in Database.__subclasses__()]
        idx = [datasets.index(subset) for subset in datasets if self.database in subset]
        parts = Database.__subclasses__()[idx[0]]().get_landmarks()
        # indices = list(itertools.chain.from_iterable(parts.values()))
        indices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 24, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 1, 134, 2, 136, 3, 138, 139, 140, 141, 4, 143, 5, 145, 6, 147, 148, 149, 150, 151, 152, 153, 17, 16, 156, 157, 158, 18, 7, 161, 9, 163, 8, 165, 10, 167, 11, 169, 13, 171, 12, 173, 14, 175, 20, 177, 178, 22, 180, 181, 21, 183, 184, 23, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197]
        for img_pred in pred.images:
            # Load image
            image = cv2.imread(img_pred.filename, cv2.IMREAD_COLOR)
            for obj_pred in img_pred.objects:
                # Squared bbox required
                bbox_width, bbox_height = obj_pred.bb[2]-obj_pred.bb[0], obj_pred.bb[3]-obj_pred.bb[1]
                max_size = max(bbox_width, bbox_height)
                shift = (float(max_size-bbox_width)*0.5, float(max_size-bbox_height)*0.5)
                bbox_squared = (obj_pred.bb[0]-shift[0], obj_pred.bb[1]-shift[1], obj_pred.bb[2]+shift[0], obj_pred.bb[3]+shift[1])
                # Enlarge bounding box
                bbox_scale = 1.6
                shift = ((max_size*bbox_scale)-max_size)*0.5
                bbox_enlarged = bbox_squared + np.array([-shift, -shift, shift, shift])
                # Project image
                T = np.zeros((2, 3), dtype=float)
                T[0, 0], T[0, 1], T[0, 2] = 1, 0, -bbox_enlarged[0]
                T[1, 0], T[1, 1], T[1, 2] = 0, 1, -bbox_enlarged[1]
                bbox_width, bbox_height = bbox_enlarged[2]-bbox_enlarged[0], bbox_enlarged[3]-bbox_enlarged[1]
                warped_image = cv2.warpAffine(image, T, (int(round(bbox_width)), int(round(bbox_height))))
                S = np.zeros((2, 3), dtype=float)
                S[0, 0], S[0, 1], S[0, 2] = self.width/bbox_width, 0, 0
                S[1, 0], S[1, 1], S[1, 2] = 0, self.height/bbox_height, 0
                warped_image = cv2.warpAffine(warped_image, S, (self.width, self.height))
                # Rescaling factor
                input_image = np.expand_dims(np.true_divide(warped_image, 255).transpose(2, 0, 1).astype(np.float32), axis=0)
                # Generate prediction
                output = self.model(torch.tensor(input_image).to(self.device))
                landmarks = np.squeeze(output.cpu().detach().numpy())
                # Save prediction
                for idx, pt in enumerate(landmarks):
                    label = indices[idx]
                    lp = list(parts.keys())[next((ids for ids, xs in enumerate(parts.values()) for x in xs if x == label), None)]
                    pt_x = pt[0] * (bbox_width/self.width) + bbox_enlarged[0]
                    pt_y = pt[1] * (bbox_height/self.height) + bbox_enlarged[1]
                    obj_pred.add_landmark(GenericLandmark(label, lp, (pt_x, pt_y), True), lps[type(lp)])
