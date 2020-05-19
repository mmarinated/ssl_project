import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
from torchvision.datasets import MNIST
import torch.optim as optim
from torchvision import transforms
import itertools

from ssl_project.vehicle_layout_prediction.data_helper import LabeledDataset
from ssl_project.data_loaders.helper import collate_fn
from ssl_project.paths import PATH_TO_DATA
from collections import Counter, defaultdict

from ssl_project.constants import LABELED_SCENE_INDEX
import os
from modelzoo import vae, autoencoder, vae_concat, mmd_vae, compute_kernel, mmd_loss_function, vae_multiple_encoders
from ssl_project.utils import compute_ats_bounding_boxes, get_bounding_boxes_from_seg
from argparse import Namespace
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.models.detection import FasterRCNN, fasterrcnn_resnet50_fpn, FasterRCNN
from simclr_transforms import *
from ssl_project.vehicle_layout_prediction.bb_utils import ProcessSegmentationMaps

class ObjectDetectionModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        assert(hparams.n_scn_train + hparams.n_scn_val + hparams.n_scn_test == len(LABELED_SCENE_INDEX) )
        self.hparams = hparams
        if (hparams.random_transform):
            self.train_transform = transforms.Compose([
                                        #transforms.RandomGrayscale(p=0.5),
                                        transforms.RandomHorizontalFlip(),
                                        get_color_distortion(s=0.5),
                                        RandomGaussianBluring(kernel_size=5),
                                        transforms.ToTensor(),
                                    ])
        else:
            self.train_transform = torchvision.transforms.ToTensor()
        self.transform = torchvision.transforms.ToTensor()

        self.n_scn_train = hparams.n_scn_train
        self.n_scn_val = hparams.n_scn_val
        self.n_scn_test = hparams.n_scn_test

        self.batch_size = hparams.batch_size
        self.learning_rate = hparams.learning_rate
        self.weight_decay = hparams.weight_decay


    def train_dataloader(self):
        dataset = LabeledDataset(
            PATH_TO_DATA, 
            f"{PATH_TO_DATA}/annotation.csv", 
            LABELED_SCENE_INDEX[:self.n_scn_train], 
            transform=self.transform,
            validation=False)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, 
            shuffle=True, num_workers=8, 
            collate_fn=collate_fn, pin_memory=True, 
            drop_last=True)

        return loader
    def val_dataloader(self):
        dataset = LabeledDataset(
            PATH_TO_DATA, 
            f"{PATH_TO_DATA}/annotation.csv",
            LABELED_SCENE_INDEX[self.n_scn_train:self.n_scn_train+self.n_scn_val], 
            transform=self.transform)
        
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, 
            shuffle=False, num_workers=8, 
            collate_fn=collate_fn, pin_memory=True)

        return loader

#    def test_dataloader(self):
#        dataset = LabeledDataset(PATH_TO_DATA, f"{PATH_TO_DATA}/annotations.csv", LABELED_SCENE_INDEX[self.n_scn_train+self.n_scn_val:], extra_info = False)
#        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=collate_fn)

    def test_dataloader(self):
        return self.val_dataloader()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.97)

        return [optimizer], [scheduler]
    
    def process_batch(self, batch):
        samples, targets, tar_sems, road_images = batch
        device = samples[0].device
        samples = torch.stack(samples).to(device)
        road_images = None#torch.stack(road_images).to(device)
        tar_sems = torch.stack(tar_sems).to(device)
        tar_sems = tar_sems > 0

        return samples, targets, tar_sems, road_images 

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    
class EncoderDecoder(ObjectDetectionModel):
    def __init__(self, hparams):
        super(EncoderDecoder, self).__init__(hparams)
        self.resnet_style = hparams.resnet_style
        self.pretrained = hparams.pretrained
        self.threshold = hparams.threshold
        self.path_to_pretrained_model = hparams.path_to_pretrained_model
        self.process_segm = ProcessSegmentationMaps()

    def forward(self, inp):
        out = self.model(inp)
        return out

    def get_threat_score(self, pred_maps, targets, threshold=None):
        if threshold is None:
            threshold = self.threshold
        
        threat_score = 0
        bbs = []
        for pred_map, target in zip(pred_maps, targets):
            bbs_k24 = self.process_segm.transform(pred_map, threshold=threshold)
            if len(bbs_k24) > 0:
                bbs_k24 = self.process_segm.convert_to_bb_space(bbs_k24, axis=-2)
            else:
                bbs_k24 = torch.zeros((1,2,4))
            ts_road_map = compute_ats_bounding_boxes(bbs_k24.cpu(), target["bounding_box"].cpu())
            threat_score += ts_road_map

        return threat_score

    def validation_epoch_end(self, outputs):
        sum_dict = sum([Counter(d) for d in outputs], Counter())
        sum_n = sum_dict.pop("n")
        avg_dict = {f"avg_{k}": v / sum_n for k, v in sum_dict.items()}
        
        if "avg_val_ts" not in avg_dict:
            avg_dict["avg_val_ts"] = 0
            
        return {"val_ts": avg_dict["avg_val_ts"], 
                "log": avg_dict, 
                "progress_bar": avg_dict}
        
#         avg_val_loss = 0
#         avg_val_ts = 0
#         n = 0
#         # ADDED
#         dict_avg = defaultdict(float)
# #         {f"AVG_{key}": 0 for key in outputs[0].keys()}
        
#         for out in outputs:
#             avg_val_loss += out["val_loss"]
#             avg_val_ts += out["val_ts"]
#             n += out["n"]
#             # ADDED
#             for key in out.keys():
#                 dict_avg[key] += out[key]

#         if n > 0:
#             avg_val_loss /= n
#             avg_val_ts /= n
#             for key in dict_avg.keys():
#                 dict_avg[key] /= n
#         else:
#             avg_val_loss = 0
#             avg_val_ts = 0
#             for key in dict_avg.keys():
#                 dict_avg[key] = 0
                
#         log_dict = {f"avg_{k}": v for k, v in log_dict.items()}
             
#         return {"val_ts": log_dict["avg_val_ts"], 
#                 "log": log_dict, 
#                 "progress_bar": log_dict}
        
class AutoEncoder(EncoderDecoder):
    def __init__(self, hparams):
        super(AutoEncoder, self).__init__(hparams)

        self.model = autoencoder(
            resnet_style=self.resnet_style, pretrained=self.pretrained,
            path_to_pretrained_model=self.path_to_pretrained_model)

        self.criterion = nn.BCELoss()

    def training_step(self, batch, batch_idx):
        samples, targets, tar_sems, road_images = self.process_batch(batch)

        pred_maps = self.model(samples)

        train_loss = self.criterion(pred_maps.squeeze(), tar_sems.float().squeeze())

        self.logger.log_metrics({"train_loss": train_loss / len(samples) }, self.global_step)
        return {"loss": train_loss, "n": len(samples)}

    def training_epoch_end(self, outputs):
        avg_training_loss = 0
        n = 0
        for out in outputs:
            avg_training_loss += out["loss"]
            n += out["n"]

        avg_training_loss /= navg_val_ts

        return {"log": {"avg_train_loss": avg_training_loss}}

    def validation_step(self, batch, batch_idx):
        samples, targets, tar_sems, road_images = self.process_batch(batch)

        pred_maps = self.model(samples)

        val_loss = self.criterion(pred_maps.squeeze(), tar_sems.float().squeeze())
        out_d = {"val_loss": val_loss, "n": len(samples) }
        
        for threshold in [0.1, 0.3, 0.5, 0.7]:
            out_d[f"val_ts_{threshold}"] = self.get_threat_score(pred_maps, targets, threshold)
        
        out_d[f"val_ts"] = out_d[f"val_ts_{0.5}"]
        
        return out_d
    
class VariationalAutoEncoder(EncoderDecoder):
    def __init__(self, hparams):
        super(VariationalAutoEncoder, self).__init__(hparams)

        self.model = vae(resnet_style=self.resnet_style, pretrained=self.pretrained,
                         path_to_pretrained_model=self.path_to_pretrained_model)

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
    
    
class VariationalAutoEncoderPretrainedHead(VariationalAutoEncoder):
    def __init__(self, hparams):
        super(VariationalAutoEncoderPretrainedHead, self).__init__(hparams)
        self.has_all_encoders = hparams.has_all_encoders
        if self.has_all_encoders:
            self.model = vae_multiple_encoders(
                resnet_style=self.resnet_style, pretrained=self.pretrained,
                path_to_pretrained_model=self.path_to_pretrained_model)
        self.is_frozen = hparams.is_frozen
        if self.is_frozen:
            if self.has_all_encoders:
                self.model.encoders.requires_grad_(False)
            else:
                self.model.encoder.requires_grad_(False)
            
        self.lr_encoder_multiplier = hparams.lr_encoder_multiplier
    
    def configure_optimizers(self):
        if self.has_all_encoders:
            ALL_EXPECTED_LAYERS = ['encoders', 'encoder_after_resnet', 'vae_decoder']

            GOT_LAYERS = [name for name, _ in self.model.named_children()]
            assert set(GOT_LAYERS) == set(ALL_EXPECTED_LAYERS), f"expected={ALL_EXPECTED_LAYERS}, got={GOT_LAYERS}"

            optimizer_for_encoder = optim.Adam(self.model.encoders.parameters(), 
               lr=self.lr_encoder_multiplier * self.learning_rate, 
               weight_decay=self.weight_decay
            )
            scheduler_for_encoder = optim.lr_scheduler.StepLR(
                                                optimizer_for_encoder, step_size=1, gamma=0.97)
            other_layers = (
                child 
                for key, child in self.model.named_children() 
                if key in ALL_EXPECTED_LAYERS[1:]
            )
        else:
            ALL_EXPECTED_LAYERS = ['encoder', 'encoder_after_resnet', 'vae_decoder']

            GOT_LAYERS = [name for name, _ in self.model.named_children()]
            assert set(GOT_LAYERS) == set(ALL_EXPECTED_LAYERS), f"expected={ALL_EXPECTED_LAYERS}, got={GOT_LAYERS}"

            optimizer_for_encoder = optim.Adam(self.model.encoder.parameters(), 
               lr=self.lr_encoder_multiplier * self.learning_rate, 
               weight_decay=self.weight_decay
            )
            scheduler_for_encoder = optim.lr_scheduler.StepLR(
                                                optimizer_for_encoder, step_size=1, gamma=0.97)
            other_layers = (
                child 
                for key, child in self.model.named_children() 
                if key in ALL_EXPECTED_LAYERS[1:]
            )
        
        optimizer = optim.Adam(itertools.chain(*[model.parameters() for model in other_layers]), 
                               lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.97)
        
        if self.is_frozen:
            return [optimizer], [scheduler]
        else:
            return [optimizer_for_encoder, optimizer], [scheduler_for_encoder, scheduler]
    
    def training_step(self, batch, batch_idx, optimizer_idx=0):
        samples, targets, tar_sems, road_images = self.process_batch(batch)

        pred_maps, mu, logvar = self.model(samples)
        train_loss, CE, KLD = self.criterion(pred_maps, tar_sems, mu, logvar)

        self.logger.log_metrics({"train_loss": train_loss / len(samples), 
                                 "train_CE": CE / len(samples), 
                                 "train_KLD": KLD / len(samples) }, 
                                self.global_step)
        return {"loss": train_loss, "n": len(samples)}
    


class MMDVariationalAutoEncoder(EncoderDecoder):
    def __init__(self, hparams):
        super(MMDVariationalAutoEncoder, self).__init__(hparams)

        self.model = mmd_vae(resnet_style=self.resnet_style, pretrained=self.pretrained,
                             path_to_pretrained_model=self.path_to_pretrained_model)

    def training_step(self, batch, batch_idx):
        samples, targets, tar_sems, road_images = self.process_batch(batch)
        b_size = samples.shape[0]
        z, pred_maps = self.model(samples)
        true_samples = torch.randn(b_size, z.shape[1]).to(samples.device)
        mmd = mmd_loss_function(true_samples, z)

        nll = (pred_maps - tar_sems.float()).pow(2).mean()
        loss = nll + mmd

        self.logger.log_metrics({"train_loss": loss / b_size }, 
                                self.global_step)
        return {"loss": loss, "n": b_size}

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
        b_size = samples.shape[0]
        z, pred_maps = self.model(samples)
        true_samples = torch.randn(b_size, z.shape[1]).to(samples.device)
        mmd = mmd_loss_function(true_samples, z)
        nll = (pred_maps - tar_sems.float()).pow(2).mean()
        val_loss = nll + mmd

        threat_score = self.get_threat_score(pred_maps, targets)

        return {"val_loss": val_loss, "val_ts": threat_score, "n": b_size }
    

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
        

class VAEFasterRCNN(ObjectDetectionModel):
    def __init__(self, hparams):
        super(VAEFasterRCNN, self).__init__(hparams)

        self.resnet_style = hparams.resnet_style
        self.threshold = hparams.threshold

        self.vae = vae(resnet_style=self.resnet_style, pretrained=False)
        #self.farcnn = fasterrcnn_resnet50_fpn(pretrained_backbone=False, num_classes=1)
        resnet_net = torchvision.models.resnet18(pretrained=False) 
        modules = list(resnet_net.children())[:-2] 
        backbone = nn.Sequential(*modules) 
        backbone.out_channels = 512
        self.farcnn = FasterRCNN(backbone=backbone, num_classes=1)
        #https://stackoverflow.com/questions/58362892/resnet-18-as-backbone-in-faster-r-cnn
        self.vae.load_state_dict(torch.load("submission2_object_detection_state_dict.pt"))

        self.criterion = self.loss_function
    def loss_function(self, pred_maps, road_images, mu, logvar):
        criterion = nn.BCELoss()
        CE = criterion(pred_maps.squeeze(), road_images.float().squeeze())
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return 0.9*CE + 0.1*KLD, CE, KLD

    def process_batch(self, batch):
        samples, targets, tar_sems, road_images = super(VAEFasterRCNN, self).process_batch(batch)
        targets_farcnn = []
        for i in range(len(samples)):
            d = {}
            bbs = targets[i]["bounding_box"]
            gt_boxes = torch.zeros((bbs.shape[0], 4)).to(bbs.device)
            for j, bb in enumerate(bbs):
                gt_boxes[j] = torch.FloatTensor([
                                                bb[0].min().item() * 10 + 400,
                                                -(bb[1].max().item() * 10) + 400,
                                                bb[0].max().item() * 10 + 400,
                                                -(bb[1].min().item() * 10) + 400
                                                ])
                #logging.info("gt_box: ", gt_boxes[j])
            d['boxes'] = gt_boxes
            d['labels'] = torch.zeros((bbs.shape[0]), dtype=torch.int64).to(bbs.device)
            targets_farcnn.append(d)
        return samples, targets, tar_sems, road_images, targets_farcnn

    def training_step(self, batch, batch_idx):
        samples, targets, tar_sems, road_images, targets_farcnn = self.process_batch(batch)
        
        pred_maps, mu, logvar, farcnn_loss = self.forward(samples, targets_farcnn)
        train_loss, CE, KLD = self.criterion(pred_maps, tar_sems, mu, logvar)

        f_loss = 0
        for key, value in farcnn_loss.items():
            f_loss += value.item()

        train_loss += f_loss
        self.logger.log_metrics({"train_loss": train_loss / len(samples), 
                                 "train_CE": CE / len(samples), 
                                 "train_KLD": KLD / len(samples),
                                 "train_farcnn_loss": f_loss }, 
                                self.global_step)

        return {"loss": train_loss, "n": len(samples)}

    def forward(self, imgs, gt_boxes = None):
        pred_maps, mu, logvar = self.vae(imgs)
        if self.training:
            rcn_out = self.farcnn(( pred_maps.unsqueeze(1) > self.threshold ).float(), gt_boxes)
        else:
            rcn_out = self.farcnn((pred_maps.unsqueeze(1) > self.threshold).float())

        return pred_maps, mu, logvar, rcn_out

    def training_epoch_end(self, outputs):
        avg_training_loss = 0
        n = 0
        for out in outputs:
            avg_training_loss += out["loss"]
            n += out["n"]

        avg_training_loss /= n

        return {"log": {"avg_train_loss": avg_training_loss}}

    def validation_step(self, batch, batch_idx):
        samples, targets, tar_sems, road_images, targets_farcnn = self.process_batch(batch)

        pred_maps, mu, logvar, pred_boxes = self.forward(samples)


        threat_score = self.get_threat_score(pred_boxes, targets)
        
        return {"val_ts": threat_score, "n": len(samples) }

    def validation_epoch_end(self, outputs):
        avg_val_ts = 0
        n = 0
        for out in outputs:
            avg_val_ts += out["val_ts"]
            n += out["n"]

        if n > 0:
            avg_val_ts /= n
        else:
            avg_val_ts = 0

        return {"val_ts": avg_val_ts, 
                "log": {"avg_val_ts": avg_val_ts}, 
                "progress_bar": {"avg_val_ts": avg_val_ts}}

    def get_threat_score(self, pred_boxes,targets):
        threat_score = 0
        for preds, target in zip(pred_boxes, targets):
            bbs_pred = preds["boxes"]
            if bbs_pred.shape[0] > 0:
                actual_boxes = torch.zeros((bbs_pred.shape[0], 2,4 ))
                for i, bb in enumerate(bbs_pred):
                    x_min, x_max = (bb[0] - 400) / 10, (bb[2] - 400) / 10
                    y_min, y_max = -(bb[1] - 400) / 10, -(bb[3] - 400) / 10
                    actual_boxes[i] = torch.FloatTensor([ [x_max, x_max, x_min, x_min],
                                                        y_max, y_min, y_max, y_min ])
                logging.info("Predicted boxes:", actual_boxes)
                logging.info("True boxes: ", target["bounding_box"] )
            else:
                actual_boxes = torch.zeros((1, 2,4 ))
                
            ts_road_map = compute_ats_bounding_boxes(actual_boxes.cpu(), target["bounding_box"].cpu())
            threat_score += ts_road_map

        return threat_score

    def configure_optimizers(self):
        optimizer = optim.Adam(list(self.vae.parameters()) + list(self.farcnn.parameters()), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.97)

        return [optimizer], [scheduler]
