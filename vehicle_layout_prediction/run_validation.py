from pl_modules import VariationalAutoEncoder, AutoEncoder, MMDVariationalAutoEncoder
from argparse import Namespace
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import numpy as np

if __name__ == "__main__":
    #path = "checkpoints/vae_final_test/checkpoint_epoch=48_val_ts=0.013.ckpt"
    hparams =  Namespace(**{"resnet_style": "18",
                          "pretrained": False,
                          "threshold": 0.3,
                          "n_scn_train": 25,
                          "n_scn_val": 3, 
                          "n_scn_test": 0,
                          "batch_size": 8,
                          "learning_rate": 0.0001,
                          "weight_decay": 0.00001,
                          "random_transform": False,
                          "path_to_pretrained_model": None})

    model = VariationalAutoEncoder(hparams)
    model.model.load_state_dict(torch.load("submission2_object_detection_state_dict.pt"))
    trainer = pl.Trainer(gpus=1)
    
    for t in np.linspace(0.43, 0.44, 20):
        print(t)
        model.threshold = t
        trainer.test(model)