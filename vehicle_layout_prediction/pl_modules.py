import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
from torchvision.datasets import MNIST
import torch.optim as optim

from ssl_project.vehicle_layout_prediction.data_helper import LabeledDataset
from ssl_project.data_loaders.helper import collate_fn
from ssl_project.paths import PATH_TO_DATA

from ssl_project.constants import LABELED_SCENE_INDEX
import os
from modelzoo import vae, autoencoder, vae_concat
from ssl_project.utils import compute_ats_bounding_boxes, get_bounding_boxes_from_seg
from argparse import Namespace
from pytorch_lightning.callbacks import ModelCheckpoint

class ObjectDetectionModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        assert(hparams.n_scn_train + hparams.n_scn_val + hparams.n_scn_test == len(LABELED_SCENE_INDEX) )
        
        self.transform = torchvision.transforms.ToTensor()

        self.n_scn_train = hparams.n_scn_train
        self.n_scn_val = hparams.n_scn_val
        self.n_scn_test = hparams.n_scn_test

        self.batch_size = hparams.batch_size
        self.learning_rate = hparams.learning_rate
        self.weight_decay = hparams.weight_decay

    def train_dataloader(self):
        dataset = LabeledDataset(PATH_TO_DATA, f"{PATH_TO_DATA}/annotation.csv", LABELED_SCENE_INDEX[:self.n_scn_train], extra_info = False, transform = self.transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)

        return loader
    def val_dataloader(self):
        dataset = LabeledDataset(PATH_TO_DATA, f"{PATH_TO_DATA}/annotation.csv", LABELED_SCENE_INDEX[self.n_scn_train:self.n_scn_train+self.n_scn_val], extra_info = False, transform = self.transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

        return loader

#    def test_dataloader(self):
#        dataset = LabeledDataset(PATH_TO_DATA, f"{PATH_TO_DATA}/annotations.csv", LABELED_SCENE_INDEX[self.n_scn_train+self.n_scn_val:], extra_info = False)
#        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=collate_fn)

    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.97)

        return [optimizer], [scheduler]
    
    def process_batch(self, batch):
        samples, targets, tar_sems, road_images = batch
        device = samples[0].device
        samples = torch.stack(samples).to(device)
        road_images = torch.stack(road_images).to(device)
        tar_sems = torch.stack(tar_sems).to(device)
        tar_sems = tar_sems > 0

        return samples, targets, tar_sems, road_images 

class EncoderDecoder(ObjectDetectionModel):
    def __init__(self, hparams):
        super(EncoderDecoder, self).__init__(hparams)
        self.resnet_style = hparams.resnet_style
        self.pretrained = hparams.pretrained
        self.threshold = hparams.threshold

    def forward(self, inp):
        out = self.model(inp)
        return out

    def get_threat_score(self, pred_maps,targets):

        threat_score = 0
        for pred_map, target in zip(pred_maps, targets):
            bb_pred = get_bounding_boxes_from_seg(pred_map > self.threshold, 10, 800, 800)
            ts_road_map = compute_ats_bounding_boxes(bb_pred.cpu(), target["bounding_box"].cpu())
            threat_score += ts_road_map

        return threat_score

    def validation_epoch_end(self, outputs):
        avg_val_loss = 0
        avg_val_ts = 0
        n = 0
        for out in outputs:
            avg_val_loss += out["val_loss"]
            avg_val_ts += out["val_ts"]
            n += out["n"]

        if n > 0:
            avg_val_loss /= n
            avg_val_ts /= n
        else:
            avg_val_loss = 0
            avg_val_ts = 0

        return {"val_ts": avg_val_ts, 
                "log": {"avg_val_loss": avg_val_loss, "avg_val_ts": avg_val_ts}, 
                "progress_bar": {"avg_val_loss": avg_val_loss, "avg_val_ts": avg_val_ts}}
        
class AutoEncoder(EncoderDecoder):
    def __init__(self, hparams):
        super(AutoEncoder, self).__init__(hparams)

        self.model = autoencoder(resnet_style=self.resnet_style, pretrained=self.pretrained)

        self.criterion = nn.BCELoss()

    def training_step(self, batch, batch_idx):
        samples, targets, tar_sems, road_images = self.process_batch(batch)

        pred_maps = model(samples)

        train_loss = self.criterion(pred_maps.squeeze(), tar_sems.float().squeeze())

        self.logger.log_metrics({"train_loss": train_loss / len(samples) }, self.global_step)
        return {"loss": train_loss, "n": len(samples)}

    def training_epoch_end(self, outputs):
        avg_training_loss = 0
        n = 0
        for out in outputs:
            avg_training_loss += out["loss"]
            n += out["n"]

        avg_training_loss /= n

        return {"log": {"avg_train_loss": avg_training_loss}}

    def validation_step(self, batch, batch_idx):
        samples, targets, tar_sems, road_images = self.process_batch(batch)

        pred_maps = model(samples)

        val_loss = self.criterion(pred_maps.squeeze(), tar_sems.float().squeeze())

        threat_score = self.get_threat_score(pred_maps, targets)

        return {"val_loss": val_loss, "val_ts": threat_score, "n": len(samples) }

class VariationalAutoEncoder(EncoderDecoder):
    def __init__(self, hparams):
        super(VariationalAutoEncoder, self).__init__(hparams)

        self.model = vae(resnet_style=self.resnet_style, pretrained=self.pretrained)

        self.criterion = self.loss_function
        self.BCE = nn.BCELoss()
    def loss_function(self, pred_maps, road_images, mu, logvar):
        criterion = nn.BCELoss()
        CE = criterion(pred_maps.squeeze(), road_images.float().squeeze())
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return 0.9*CE + 0.1*KLD, CE, KLD

    def training_step(self, batch, batch_idx):
        samples, targets, tar_sems, road_images = self.process_batch(batch)

        pred_maps, mu, logvar = self.model(samples)
        train_loss, CE, KLD = self.criterion(pred_maps, tar_sems, mu, logvar)

        self.logger.log_metrics({"train_loss": train_loss / len(samples), 
                                 "train_CE": CE / len(samples), 
                                 "train_KLD": KLD / len(samples) }, 
                                self.global_step)
        return {"loss": train_loss, "n": len(samples)}

    def training_epoch_end(self, outputs):
        avg_training_loss = 0
        n = 0
        for out in outputs:
            avg_training_loss += out["loss"]
            n += out["n"]

        avg_training_loss /= n

        return {"log": {"avg_train_loss": avg_training_loss}}

    def validation_step(self, batch, batch_idx):
        samples, targets, tar_sems, road_images = self.process_batch(batch)

        pred_maps, mu, logvar = self.model(samples)

        val_loss = self.BCE(pred_maps.squeeze(), tar_sems.float().squeeze())

        threat_score = self.get_threat_score(pred_maps, targets)

        return {"val_loss": val_loss, "val_ts": threat_score, "n": len(samples) }

class VariationalAutoEncoderConcat(EncoderDecoder):
    def __init__(self, hparams):
        super(VariationalAutoEncoderConcat, self).__init__(hparams)

        self.model = vae_concat(resnet_style=self.resnet_style, pretrained=self.pretrained)

        self.criterion = self.loss_function
        self.BCE = nn.BCELoss()
    def loss_function(self, pred_maps, road_images, mu, logvar):
        criterion = nn.BCELoss()
        CE = criterion(pred_maps.squeeze(), road_images.float().squeeze())
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return 0.9*CE + 0.1*KLD, CE, KLD

    def training_step(self, batch, batch_idx):
        samples, targets, tar_sems, road_images = self.process_batch(batch)

        pred_maps, mu, logvar = model(samples)
        train_loss, CE, KLD = self.criterion(pred_maps, tar_sems, mu, logvar)

        self.logger.log_metrics({"train_loss": train_loss / len(samples), 
                                 "train_CE": CE / len(samples), 
                                 "train_KLD": KLD / len(samples) }, 
                                self.global_step)
        return {"loss": train_loss, "n": len(samples)}

    def training_epoch_end(self, outputs):
        avg_training_loss = 0
        n = 0
        for out in outputs:
            avg_training_loss += out["loss"]
            n += out["n"]

        avg_training_loss /= n

        return {"log": {"avg_train_loss": avg_training_loss}}

    def validation_step(self, batch, batch_idx):
        samples, targets, tar_sems, road_images = self.process_batch(batch)

        pred_maps, mu, logvar = model(samples)

        val_loss = self.BCE(pred_maps.squeeze(), tar_sems.float().squeeze())

        threat_score = self.get_threat_score(pred_maps, targets)

        return {"val_loss": val_loss, "val_ts": threat_score, "n": len(samples) }
        
