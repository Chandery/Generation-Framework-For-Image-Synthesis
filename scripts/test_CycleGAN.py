import hydra
import lightning as L
import os
import sys
from lightning.pytorch.callbacks import ModelCheckpoint

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from models.CycleGAN import CycleGAN
from datasets.unaligned_dataset import UnalignedDataset
from torch.utils.data import DataLoader

@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(config):
    config = config["config"]
    
    test_ds = UnalignedDataset(opt=config.DatasetConfig, split="test")
    
    test_dl = DataLoader(test_ds, 
                        batch_size=1, 
                        shuffle=True, 
                        num_workers=config.DatasetConfig.num_workers)
    
    model = CycleGAN(**config["CycleGAN"], 
                     is_Train=False, 
                     root_dir=config.hydra_path, 
    )
    
    trainer = L.Trainer(**config["trainer"], default_root_dir=config.hydra_path)
    model.init_from_ckpt(config.ckpt)
    
    trainer.test(model=model, dataloaders=test_dl)

if __name__ == "__main__":
    main()