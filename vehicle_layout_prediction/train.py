from pl_modules import VariationalAutoEncoder, AutoEncoder, MMDVariationalAutoEncoder
from argparse import Namespace
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


if __name__ == "__main__":
    hparams =  Namespace(**{"resnet_style": "18",
                          "pretrained": False,
                          "threshold": 0.4,
                          "n_scn_train": 24,
                          "n_scn_val": 3, 
                          "n_scn_test": 1,
                          "batch_size": 8,
                          "learning_rate": 0.0001,
                          "weight_decay": 0.00001,
                          "random_transform": False})

    experiment_name = "vae_with_blur"

    checkpoint_callback = ModelCheckpoint(
                        filepath="./checkpoints/" + experiment_name + "/checkpoint_{epoch}_{val_ts:.6f}",
                        save_top_k=5,
                        verbose=False,
                        monitor='val_ts',
                        mode='max',
                        prefix=''
                    )
    print(f"Experiment: {experiment_name}")
    print("Parameters...")
    print(hparams)

    logger = pl.loggers.TensorBoardLogger("tb_logs", experiment_name)
    logger.log_hyperparams(hparams)
    model = VariationalAutoEncoder(hparams)
    trainer = pl.Trainer(logger=logger, gpus=1, max_epochs=100, checkpoint_callback=checkpoint_callback)

    trainer.fit(model)