import random
import argparse

from utils.hpo import HyperParameterOptimizer as HPO
from utils.utils import load_config
from utils.Trainer import TorchTrainer

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

project_name = ""  # project name for logging


def run_training():
    # Train the model.

    # set dataloader
    dataloader = TextDataloader(
        model_name=config.model_name,
        batch_size=config.training.batch_size,
        shuffle=config.training.shuffle,
        normalization = config.normalization,
        train_path=config.training.train_path,
        dev_path=config.test.dev_path,
        test_path=config.test.test_path,
        predict_path=config.test.predict_path
    )
    dataloader.setup(stage="fit")
    train_loader = dataloader.train_dataloader()
    val_loader = dataloader.val_dataloader()

    trainer = TorchTrainer(config, use_wandb=wandb.run)
    
    loss, pearson = trainer.train(train_loader, val_loader)

    return loss, pearson

def run_sweep():
    # Target function for wandb.sweep
    with wandb.init() as run:
        set_config(config, wandb.config)
        run_training()

def run():
    # Train the model.

    # 2 different hpo APIs
    match config.training.hpo:
        case "wandb_sweep":
            # You must check that the args.wandb is 'True' 
            sweep_config = init_sweep_config()    
            sweep_id = wandb.sweep(sweep_config, project=project_name)
            wandb.agent(sweep_id, function=run_sweep)
        case "optuna":
            hpo = HPO(100, config)
            hpo.set_project_name(project_name)
            hpo.set_sampler("TPE")
            hpo.set_pruner("Hyperband")
            hpo.optimize(direction="minimize")
        case _:
            run_training()

def init_sweep_config():
    # Initialize the config for wandb.sweep to search the hyperparameters.
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

    return sweep_config

def set_config(config, new_config):
    # new_config에서 하이퍼파라미터 가져오기
    config.training.batch_size = new_config.batch_size
    config.training.max_epoch = new_config.max_epoch
    config.training.learning_rate = new_config.learning_rate
    config.training.optimizer.weight_decay = new_config.weight_decay
    config.training.scheduler.patience = new_config.scheduler_patience


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Specify your config file name in ./config/")
    parser.add_argument("--wandb", action="store_true", help="Use W&B visualization") # Wandb DashBoard 활성화
    args = parser.parse_args()

    config_path = f"./config/{args.config}.yaml"
    config = load_config(config_path)

    if args.wandb:
        # W&B enabled
        os.environ["WANDB_API_KEY"] = ""  # "(본인 키 입력)" 
        project_name = f"project1_sts_test_{config.model_name.split('/')[1]}"

        wandb.login()
    else:
        # W&B disabled
        wandb.init(mode="disabled")

    run()

