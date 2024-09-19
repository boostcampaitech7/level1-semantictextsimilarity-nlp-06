import random
import argparse

from utils.utils import load_config
from utils.Trainer import torch_Trainer

import torch
from dataset.dataloader import TextDataloader


# seed 고정
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="Base_config", required=True)
    args = parser.parse_args()

    config_path = f"./config/{args.config}.yaml"
    config = load_config(config_path)

    dataLoader = TextDataloader(
                    model_name=config.model_name,
                    batch_size=config.training.batch_size,
                    shuffle=config.training.shuffle,
                    train_path=config.training.train_path,
                    dev_path=config.test.dev_path,
                    test_path=config.test.test_path,
                    predict_path=config.test.predict_path
                )
    dataLoader.setup(stage="fit")
    train_loader = dataLoader.train_dataloader()
    val_loader = dataLoader.val_dataloader()
    
    trainer = torch_Trainer(config)
    trainer.train(train_loader, val_loader)
    
    

    