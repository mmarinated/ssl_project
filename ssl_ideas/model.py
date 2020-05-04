import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import pytorch_lightning as pl
from ssl_project.road_layout_prediction.modelzoo import encoder
from torch.utils.data import DataLoader

from ssl_project.constants import LABELED_SCENE_INDEX, UNLABELED_SCENE_INDEX


train_idces = UNLABELED_SCENE_INDEX
val_idces = LABELED_SCENE_INDEX



class ShuffleAndLearnModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.resnet_encoder = encoder(resnet_style='18', pretrained=False)
        OUT_ENC_CHANNELS = 512 # might change in the future
        # output 512 x 8 x 8
        self.decoder = nn.Sequential(
            nn.Conv2d(3 * OUT_ENC_CHANNELS, OUT_ENC_CHANNELS, kernel_size=(3, 3), stride=(1, 1), bias=False),
            nn.BatchNorm2d(OUT_ENC_CHANNELS, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )
        self.clf = nn.Linear(in_features=OUT_ENC_CHANNELS, out_features=2, bias=True)
        self.criterion = nn.BCEWithLogitsLoss(reduction="sum")
        self.threshold = 0.5
        
#     def forward(self, images_bo3hw, cam_name):
#         resnet_encoder = self.cam_name_to_encoder[cam_name]
#         image_1_bcll = resnet_encoder(image_bo3hw[:, 0])
#         image_2_bcll = resnet_encoder(image_bo3hw[:, 1])
#         image_3_bcll = resnet_encoder(image_bo3hw[:, 2])
        
#         out_b2 = self.classifier(torch.cat((image_1_bcll, image_2_bcll, image_3_bcll), dim=1))
#         return out_b2
    
        
    def forward(self, images_bo3hw):
        # c = 512, l = 8
        image_1_bcll = self.resnet_encoder(images_bo3hw[:, 0])
        image_2_bcll = self.resnet_encoder(images_bo3hw[:, 1])
        image_3_bcll = self.resnet_encoder(images_bo3hw[:, 2])
        out_bm11 = self.decoder(torch.cat((image_1_bcll, image_2_bcll, image_3_bcll), dim=1))
#         assert out_bm11.shape[-2:] == (1, 1)
        out_b2 = self.clf(out_bm11.view(out_bm11.shape[:2]))
        return out_b2
    
    def training_step(self, batch, batch_idx):
        images_bo3hw, labels_b = batch
        out_b = self.forward(images_bo3hw)[:, 1]
        loss = self.criterion(out_b, labels_b.type_as(out_b))
        tensorboard_logs = {'train_loss': loss}
        return {
            'loss': loss, 
            'log': tensorboard_logs
        }
    
    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def train_dataloader(self):
        # REQUIRED
        return DataLoader(TripleDataset(scene_ids=train_idces), num_workers=4, batch_size=32)

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        images_bo3hw, labels_b = batch
        out_b = self.forward(images_bo3hw)[:, 1]
        return {
            'val_loss': self.criterion(out_b, labels_b.type_as(out_b)),
            'val_acc' : ((out_b > self.threshold).long() == labels_b).float().sum(),
        }
    
    def validation_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).sum() / self.val_size
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).sum() / self.val_size
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc' : avg_acc,}
        return {
            'val_loss': avg_loss, 
            'val_acc' : avg_acc,
            'log': tensorboard_logs
        }
 
    @property
    def val_size(self):
        return len(self.val_dataloader().dataset)

    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(TripleDataset(scene_ids=val_idces), num_workers=4, 
                          batch_size=32, shuffle=False, pin_memory=True)