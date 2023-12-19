#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Juan Castrillo'
__email__ = 'juan.castrillo@alumnos.upm.es'

import os
import cv2
import torch
import dlib
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from images_framework.src.alignment import Alignment
from .models.get_model import get_model
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
        self.database = "WFLW"

    '''
    model
        ConvNeXt 
            convnext_atomic
        MobileNets
            mobilenetv2
        MobileViTs
            EdgeNeXt
                edgenext_small
                edgenext_custom_a
                ...
            EfficientFormer
                efficientformerv2_(L, S1, S2, S0)
    '''
    def parse_options(self, params):
        super().parse_options(params)
        import argparse
        parser = argparse.ArgumentParser(prog='ICITS24Landmarks', add_help=False)
        
        parser.add_argument('--model', dest='model_name', type=str, action='append',
                            help='Model name from the list.')
        parser.add_argument('--gpu', dest='gpu', type=int, action='append', default=None,
                            help='GPU number to use.')
        parser.add_argument('--batch_size', dest='batch_size', type=int, action='append',
                            help='Batch Size to use for training and inference.')
        parser.add_argument('--epochs', dest='epochs', type=int, action='append',
                            help='Number of epochs to train for.')
        parser.add_argument('--patience', dest='patience', type=int, action='append',
                            required=False, default=10,
                            help='Early stopping patience')
        parser.add_argument('--trained_model', dest='trained_model', type=str, action='append',
                            required=False, default=None,
                            help='File containing the trained model.')
        parser.add_argument('--output_path', dest='output_path', type=str, action='append',
                            required=False, default=None,
                            help='Path for all neccesary model outputs.')


        args, unknown = parser.parse_known_args(params)
        print(parser.format_usage())
        self.gpu = args.gpu
        self.patience = args.patience
        self.epochs = args.epochs
        self.batch_size = args.batch_size

        self.model_name = args.model_name
        self.model = self.get_model(self.model_name)
        self.trained_model_name = args.trained_model

        self.output_path = args.output_path
        

    '''
    Train model on WFLW (98 landmarks)
    TODO - Is it one epoch? or the whole set, if the later how to handle the images.
        Asumo que son dataloaders para todo.
        anns_train: [{image: ... , landmarks: [[x_1,y_1], [], ...]}, {...}, ...]
    '''
    def train(self, anns_train, anns_valid):
        accelerator = 'gpu' if self.gpu != None else 'cpu'
        precision = 16 #if config['amp'] else 32

        loggers = [pl_loggers.TensorBoardLogger(save_dir=self.output_path+"/logs", default_hp_metric=False),]  # Tensorboard logger in <exp_dir>/logs/ # Metrics logger

        checkpoint_callback = ModelCheckpoint(dirpath=self.output_path+'/ckpt_path',
                                          filename='{epoch}-{val_loss:.5f}',
                                          monitor='val_loss',
                                          save_last=True,
                                          save_top_k=1)

        early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=self.patience)


        trainer = pl.Trainer(logger=loggers,
                         accelerator=accelerator,                          # CPU or GPU
                         devices=self.gpu,                                # GPU ids
                         enable_progress_bar=True,                        # Do not show progress bar
                         max_epochs=self.epochs,                      # Max number of epochs
                         precision=precision,                              # float16 if AMP is enabled, else float32
                         deterministic=False,                               # Deterministic behavior
                        #  gradient_clip_val=config['gradient_clip'],        # Gradient clip value
                         callbacks=[checkpoint_callback, early_stopping])

        trainer.fit(model=self.model, train_dataloaders=anns_train, val_dataloaders=anns_valid, )#ckpt_path=resume_path)
    


    '''
    Load pretrained model
    as set in the arguments.
    '''
    def load(self, mode):
        from images_framework.src.constants import Modes
        print('Loading model...')
        if mode is Modes.TEST:
            self.saved_model = self.path + 'data/' + self.database + '/' + self.trained_model_name + '.pt'
            model = torch.load(self.saved_model, map_location="cpu")
        else:
            if self.trained_model_name:
                self.saved_model = self.path + 'data/' + self.database + '/' + self.trained_model_name + '.pt'
                model = torch.load(self.saved_model)#, map_location="cpu")
            else:
                model = get_model(self.model_name)
        return model
    
    # TODO - Add method to export to mobile
    def export_to_mobile(self):
        self.model.eval()
        example = torch.rand(1, 3, 128, 128)
        traced_script_module = torch.jit.trace(self.model, example)
        traced_script_module_optimized = optimize_for_mobile(traced_script_module)
        traced_script_module_optimized._save_for_lite_interpreter("app/src/main/assets/model.ptl")


    # TODO - Add notebook to evaluate latency

    # TODO - Ni idea de como tratar esto, tendré que ver como cuadrar ids de landmarks a la imagen.
    def process(self, ann, pred):
        '''
        
        ann: {id1: (x1, y1), ... }
        pred: {}
        '''
        import itertools
        from src.datasets import Database
        from src.annotations import GenericLandmark
        from alignment.landmarks import lps

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
