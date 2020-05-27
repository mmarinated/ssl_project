from argparse import Namespace
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import pytorch_lightning as pl
from ssl_project.road_layout_prediction.modelzoo import encoder
from torch.utils.data import DataLoader

from ssl_project.constants import LABELED_SCENE_INDEX, UNLABELED_SCENE_INDEX, CAM_NAMES

from ssl_project.ssl_ideas.preprocessing import TripleDataset

train_idces = UNLABELED_SCENE_INDEX
val_idces = LABELED_SCENE_INDEX


def SET_SEED(seed=57):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    

class ShuffleAndLearnNet(nn.Module):
    def __init__(self, fit_all_encoders):
        """
        Parameters
        ----------
        fit_all_encoders
        """
        super().__init__()
        # compute hparams
        
        self.fit_all_encoders = fit_all_encoders
        
        self.resnet_style = '18'
        self.pretrained   = False
        
        if self.fit_all_encoders:
            self.cam_name_to_encoder = torch.nn.ModuleDict({
                cam_name : encoder(resnet_style=self.resnet_style, pretrained=self.pretrained)
                for cam_name in CAM_NAMES
            })
        else:
            self.resnet_encoder = encoder(resnet_style=self.resnet_style, pretrained=self.pretrained)

        OUT_ENC_CHANNELS = 512 # might change in the future
        # output 512 x 8 x 8
        self.decoder = nn.Sequential(
            nn.Conv2d(3 * OUT_ENC_CHANNELS, OUT_ENC_CHANNELS, kernel_size=(3, 3), stride=(1, 1), bias=False),
            nn.BatchNorm2d(OUT_ENC_CHANNELS, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )
        self.clf = nn.Linear(in_features=OUT_ENC_CHANNELS, out_features=1, bias=True)
                
    def encode_3_images(self, encoder, images_bo3hw):
        image_1_bcll = encoder(images_bo3hw[:, 0])
        image_2_bcll = encoder(images_bo3hw[:, 1])
        image_3_bcll = encoder(images_bo3hw[:, 2]) 
        return torch.cat((image_1_bcll, image_2_bcll, image_3_bcll), dim=1)
        
    def forward(self, images_bo3hw, cam_names_b):
        n_l = 8
        n_c = 512
        n_C = n_c * 3
        if not self.fit_all_encoders: 
            # easy version
            images_bCll = self.encode_3_images(self.resnet_encoder, images_bo3hw)
        else:
            # create output and run for loop
            images_bCll = torch.zeros((images_bo3hw.shape[0], n_C, n_l, n_l), 
                                      dtype=images_bo3hw.dtype, device=images_bo3hw.device) # ???
            for cam_name, encoder in self.cam_name_to_encoder.items():
                b_slc = np.isin(cam_names_b, cam_name)
                if b_slc.any() > 0:
                    images_bCll[b_slc] = self.encode_3_images(encoder, images_bo3hw[b_slc])                

                    
                    
        out_bm11 = self.decoder(images_bCll)
        out_b1 = self.clf(out_bm11.view(out_bm11.shape[:2]))
        return out_b1

    
    
    

class ShuffleAndLearnModel(pl.LightningModule):
    def __init__(self, hparams):
        """
        - fit_all_encoders
        - lr
        - num_workers
        """
        super().__init__()
        self.hparams = hparams
        
        self.model = ShuffleAndLearnNet(fit_all_encoders=self.hparams.fit_all_encoders)        
        self.criterion = nn.BCEWithLogitsLoss(reduction="mean")
        self.threshold = 0.5
        self.num_workers = self.hparams.num_workers
        
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    
    ###
    ## Loaders
    ###
    
    def train_dataloader(self):
        return DataLoader(
            TripleDataset(cam_names=CAM_NAMES, scene_ids=train_idces), 
            num_workers=self.num_workers, 
            batch_size=self.hparams.batch_size,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            TripleDataset(cam_names=CAM_NAMES, scene_ids=val_idces), 
            num_workers=self.num_workers, 
            batch_size=self.hparams.batch_size, 
            shuffle=False, 
            pin_memory=True
        )
    
    
    ###
    ## Training
    ###
    
        
    def forward(self, images_bo3hw, cam_names_b):
        return self.model(images_bo3hw, cam_names_b)

        
    def training_step(self, batch, batch_idx):
        images_bo3hw, cam_names_b, labels_b = batch
        out_b = self.forward(images_bo3hw, cam_names_b).view(-1)
        loss = self.criterion(out_b, labels_b.type_as(out_b))
        tensorboard_logs = {'train_loss': loss}
        return {
            'loss': loss, 
            'log': tensorboard_logs
        }

    
    def validation_step(self, batch, batch_idx):
        images_bo3hw, cam_names_b, labels_b = batch
        out_b = self.forward(images_bo3hw, cam_names_b).view(-1)
        probs_b = torch.sigmoid(out_b)
        return {
            'val_loss': self.criterion(out_b, labels_b.type_as(out_b)),
            'val_acc' : ((probs_b > self.threshold).long() == labels_b).float().mean(),
        }
    
    ###
    ## Logging and metrics
    ###
    
    def on_train_start(self):
        if hasattr(self, "logger") and self.logger is not None:
            self.logger.log_hyperparams_metrics(
                self.hparams, {'val_loss': 1, 'val_acc': 0})

    
    def validation_epoch_end(self, outputs):
        avg_loss = (torch.stack([x['val_loss'] for x in outputs])).mean().item()
        avg_acc = (torch.stack([x['val_acc'] for x in outputs])).mean().item()

        tensorboard_logs = {'val_loss': avg_loss, 'val_acc' : avg_acc,}
                
        return {
            'val_loss': avg_loss, 
            'val_acc' : avg_acc,
            'log': tensorboard_logs
        }
