import random
import argparse

from utils.utils import load_config
from utils.Trainer import torch_Trainer

import torch
from dataset.dataloader import TextDataloader
import wandb
import os

# seed 고정
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)

def run_training(config):
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

    trainer = torch_Trainer(config, use_wandb=args.wandb_sweep)
    trainer.train(train_loader, val_loader)


def sweep_train():
    with wandb.init() as run:
        # wandb.config에서 하이퍼파라미터 가져오기
        config.training.batch_size = wandb.config.batch_size
        config.training.max_epoch = wandb.config.max_epoch
        config.training.learning_rate = wandb.config.learning_rate
        config.training.optimization.weight_decay = wandb.config.weight_decay
        config.training.scheduler.patience = wandb.config.scheduler_patience

        run_training(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="Base_config", required=True)
    parser.add_argument("--wandb_sweep", action="store_true", help="Use W&B sweep for hyperparameter tuning")
    args = parser.parse_args()

    config_path = f"./config/{args.config}.yaml"
    config = load_config(config_path)

    if args.wandb_sweep:
        sweep_config = {
            'method': 'bayes',
            'metric': {
                'name': 'validation_pearson',
                'goal': 'maximize'
            },
            'parameters': {
                'batch_size': {'values': [4, 8, 16]},
                'max_epoch': {'values': [5, 10, 20, 30, 40]},
                'learning_rate': {
                    'distribution': 'log_uniform',
                    'min': 1e-6,
                    'max': 1e-4
                },
                'weight_decay': {
                    'distribution': 'uniform',
                    'min': 0.001,
                    'max': 0.1
                },
                'scheduler_patience': {'values': [3, 5, 7]}
            }
        }

        os.environ["WANDB_API_KEY"] = "(본인 키 입력)"  ##TODO: 팀으로도 가능할 것 같은데, 추후 추가
        wandb.login()
        
        sweep_id = wandb.sweep(sweep_config, project=f"project1_sts_test_{config.model_name.split('/')[1]}")
        wandb.agent(sweep_id, function=sweep_train)
    else:
        # W&B 없이 기본 설정으로 실행
        run_training(config)

