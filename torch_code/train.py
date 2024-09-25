import random
import argparse

from utils.utils import load_config
from utils.Trainer import torch_Trainer

import torch
from dataset.dataloader import TextDataloader
import wandb
import os

# seed 고정
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)

def run_training(config):
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
                'name': 'validation_loss',
                'goal': 'minimize'
            },
            'parameters': {
                'batch_size': {'values': [16]},
                'max_epoch': {'values': [10]},
                'learning_rate': {
                    'distribution': 'uniform',
                    'min': 1e-5,
                    'max': 4e-5
                },
                'weight_decay': {
                    'distribution': 'uniform',
                    'min': 0.001,
                    'max': 0.1
                }
            },
            'early_terminate': {  # sweep의 조기 종료 옵션 설정
                'type': 'hyperband',
                's': 2,  # Hyperband의 bracket 수를 결정합니다. 높은 값일수록 더 많은 bracket이 생성
                'eta': 3,  # 리소스 할당 비율 (예: eta=3이면 각 단계에서 상위 1/3만 남김)
                'max_iter':25  # 각 구성에 대해 최대 반복 횟수
            }
        }

        os.environ["WANDB_API_KEY"] = "0a7cec09004c2912bda4fd4f050dbd836efab943" 
        wandb.login()
        
        sweep_id = wandb.sweep(sweep_config, project=f"project1_sts_test_{config.model_name.split('/')[1]}")
        wandb.agent(sweep_id, function=sweep_train)
    else:
        # W&B 없이 기본 설정으로 실행
        run_training(config)

