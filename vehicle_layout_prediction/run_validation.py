from pl_modules import VariationalAutoEncoder, AutoEncoder, MMDVariationalAutoEncoder
from argparse import Namespace
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

if __name__ == "__main__":
    path = "checkpoints/vae_final_test/checkpoint_epoch=48_val_ts=0.013.ckpt"
    hparams =  Namespace(**{"resnet_style": "18",
                          "pretrained": False,
                          "threshold": 0.3,
                          "n_scn_train": 24,
                          "n_scn_val": 3, 
                          "n_scn_test": 1,
                          "batch_size": 8,
                          "learning_rate": 0.0001,
                          "weight_decay": 0.00001})
    model = VariationalAutoEncoder(hparams)
    model.load_state_dict(torch.load(path, map_location="cpu")['state_dict'])
    trainer = pl.Trainer()
    
    torch.save(model.model.state_dict(), "submission2_object_detection_state_dict.pt")
    for t in [0.3,0.4,0.5]:
        print(t)
        model.threshold = t
        trainer.test(model)