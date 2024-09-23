import random
import argparse

from utils.utils import load_config
from utils.Trainer import torch_Trainer

import pandas as pd
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
    parser.add_argument("--saved_model", required=True)
    args = parser.parse_args()

    config_path = f"./config/{args.config}.yaml"
    config = load_config(config_path)

    dataLoader = TextDataloader(
                    model_name=config.model_name,
                    batch_size=config.training.batch_size,
                    shuffle=config.training.shuffle,
                    normalization = config.normalization,
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
    
    if config.normalization:
        predictions = list(round(float(i)*5, 1) for i in predictions)  # 실제 label 범위로 원복
    else:
        predictions = list(round(float(i), 1) for i in predictions)


    # 폴더 만드는 것 까지
    output = pd.read_csv("../../data/sample_submission.csv")
    output["target"] = predictions
    output.to_csv('./output/output.csv', index=False)
    print("Complete Extract ouptut.csv")