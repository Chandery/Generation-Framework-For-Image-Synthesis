import hydra
import lightning as L
import os
import sys
from lightning.pytorch.callbacks import ModelCheckpoint

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from models.CycleGAN import CycleGAN
from datasets.unaligned_dataset import UnalignedDataset
from torch.utils.data import DataLoader
import torch

torch.set_float32_matmul_precision('high')

@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(config):
    config = config["config"]
    config.trainer.max_epochs = config.train.n_epochs + config.train.n_epoch_decay
    filename = f"train_CycleGAN-epoch{{epoch:02d}}-{config.monitor.replace('/', '_')}={{{config.monitor}:.2f}}"
    print("save model to", filename)
    
    checkpoint_callback = ModelCheckpoint(
        monitor=config.monitor,
        dirpath=config.hydra_path,
        filename=filename,
        save_top_k=1,
        mode="min",
        auto_insert_metric_name=False,
        save_last=True,
    )
    
    train_ds = UnalignedDataset(opt=config.DatasetConfig, split="train")
    val_ds = UnalignedDataset(opt=config.DatasetConfig, split="val")
    
    print("train_ds size:", len(train_ds))
    print("val_ds size:", len(val_ds))
    
    train_dl = DataLoader(train_ds, 
                        batch_size=config.DatasetConfig.batch_size, 
                        shuffle=True, 
                        num_workers=config.DatasetConfig.num_workers)
    val_dl = DataLoader(val_ds, 
                      batch_size=config.DatasetConfig.batch_size, 
                      shuffle=False, 
                      num_workers=config.DatasetConfig.num_workers)

    val_batch_total = len(val_ds) // config.DatasetConfig.batch_size
    model = CycleGAN(**config["CycleGAN"], 
                     is_Train=True, 
                     root_dir=config.hydra_path, 
                     val_batch_total=val_batch_total
    )
    
    trainer = L.Trainer(**config["trainer"], callbacks=[checkpoint_callback], default_root_dir=config.hydra_path)
    
    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)

if __name__ == "__main__":
    main()