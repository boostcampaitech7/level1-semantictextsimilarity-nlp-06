import random
import argparse

from utils.utils import load_config
from utils.Trainer import torch_Trainer

import pandas as pd
import torch
from dataset.dataloader import TextDataloader

import os

# seed 고정
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Specify your config file name in ./config/")
    parser.add_argument("--saved_model", required=True, help="Specify your saved model name")
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
    dataLoader.setup(stage="test")
    predict_loader = dataLoader.predict_dataloader()
    test_loader = dataLoader.test_dataloader()
    
    trainer = torch_Trainer(config)
    model = torch.load(f"./saved_model/{args.saved_model}.pt")
    predictions = trainer.predict(model=model, dataloader=predict_loader)
    predictions = list(round(float(i), 1) for i in predictions)

    # output.csv 저장할 경로
    folder_path = './output/'

    # 폴더가 존재하지 않으면 생성
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    output = pd.read_csv("../data/sample_submission.csv")
    output["target"] = predictions
    output.to_csv('./output/output.csv', index=False)
    print("Complete Extract ouptut.csv")