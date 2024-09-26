import argparse
from configurer import configurer
from torch.utils.data import DataLoader

from dataset import preprocess, Dataset
from model import Model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config.yaml", help="path of the yaml file")
    arg = parser.parse_args()

    config = configurer(arg.config)

    train_dataset = preprocess(task="train", data_path=config.train_path, model_name=config.model, scale=config.scale)
    valid_dataset = preprocess(task="valid", data_path=config.valid_path, model_name=config.model, scale=config.scale)

    model = Model(sweeps=config.sweeps,
                  model_name=config.model,
                  lora=config.lora,
                  output_dir=config.output_dir,
                  epoch=config.epoch,
                  train_data=train_dataset,
                  valid_data=valid_dataset,
                  batch_size=config.batch_size,
                  lr=float(config.lr_init),
                  lr_scheduler=config.lr_scheduler,
                  weight_decay=config.weight_decay)