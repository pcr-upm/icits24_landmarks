#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Juan Castrillo'
__email__ = 'juan.castrillo@alumnos.upm.es'

import os
import cv2
import dlib
import numpy as np
from images_framework.src.alignment import Alignment
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)


class ICITS24Landmarks(Alignment):
    """
    Object alignment using ICITS 2024 algorithm
    """
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.model = None

    def parse_options(self, params):
        super().parse_options(params)
        import argparse
        parser = argparse.ArgumentParser(prog='ICITS24Landmarks', add_help=False)
        parser.add_argument('--gpu', dest='gpu', type=int, action='append',
                            help='GPU ID (negative value indicates CPU).')
        args, unknown = parser.parse_known_args(params)
        print(parser.format_usage())
        self.gpu = args.gpu

    def train(self, anns_train, anns_valid):
        print('Train model')

    def load(self, mode):
        from images_framework.src.constants import Modes
        # Set up a neural network to train
        print('Load model')
        if mode is Modes.TEST:
            dat_file = self.path + 'data/' + self.database + '/' + self.database + '_6.dat'
            self.model = dlib.shape_predictor(dat_file.lower())

    def process(self, ann, pred):
        import itertools
        from images_framework.src.datasets import Database
        from images_framework.src.annotations import GenericLandmark
        from images_framework.alignment.landmarks import lps
        datasets = [subclass().get_names() for subclass in Database.__subclasses__()]
        idx = [datasets.index(subset) for subset in datasets if self.database in subset]
        parts = Database.__subclasses__()[idx[0]]().get_landmarks()
        indices = list(itertools.chain.from_iterable(parts.values()))
        for img_pred in pred.images:
            # Load image
            image = cv2.imread(img_pred.filename)
            for obj_pred in img_pred.objects:
                # Generate prediction
                rect = dlib.rectangle(int(round(obj_pred.bb[0])), int(round(obj_pred.bb[1])), int(round(obj_pred.bb[2])), int(round(obj_pred.bb[3])))
                shape = self.model(image, rect)
                # Save prediction
                for idx, pt in enumerate(shape.parts()):
                    label = indices[idx]
                    lp = list(parts.keys())[next((ids for ids, xs in enumerate(parts.values()) for x in xs if x == label), None)]
                    obj_pred.add_landmark(GenericLandmark(label, lp, (pt.x, pt.y), True), lps[type(lp)])
