from utils.utils import ckpt_save
import pandas as pd

from tqdm.auto import tqdm

import transformers
import torch
import torchmetrics
import pytorch_lightning as pl
import wandb


# 일정 epoch동안 성능 개선 없으면 조기 종료
class EarlyStopping:
    def __init__(self, patience=5, delta=0.0001):
        self.patience = patience  # 성능 개선 없이 허용되는 에폭 수
        self.delta = delta  # 성능 개선으로 간주되는 최소한의 변화량
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score >= self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class TorchTrainer():
    def __init__(self, config, use_wandb=False):

        self.model_name = config.model_name
        self.batch_size = config.training.batch_size
        self.learning_rate = config.training.learning_rate
        self.criterion = config.training.criterion
        self.scheduler = config.training.scheduler
        self.optimizer = config.training.optimizer
        self.max_epoch = config.training.max_epoch
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.early_stop = config.training.early_stop
        self.use_wandb = use_wandb
        self.hpo = config.training.hpo
        self.trial = None

    def set_trial(self, trial):
        self.trial = trial

    def get_finetuned_model(self):
        return self.model
    
    def get_model(self, model_name):
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, num_labels=1
        )
        return model
        # if you want change custom model, redefine this function

    def get_optimizer(self, model, optimizer):
        if optimizer.name == "AdamW":
            optim = torch.optim.AdamW(model.parameters(), weight_decay=optimizer.weight_decay,
                                      lr=self.learning_rate)
        # Add Optim

        return optim

    def get_criterion(self, criterion):
        # Add criterion(loss function)
        if criterion == "MSELoss":
            return torch.nn.MSELoss()
            
        elif criterion == "L1Loss":
            return torch.nn.L1Loss()
            
        elif criterion == "HuberLoss":
            return torch.nn.HuberLoss()

    
    def get_scheduler(self, optimizer, scheduler, verbose):
        # LR Scheduler
        if scheduler.name == "ReduceLROnPlateau":
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                              mode="min",
                                                              patience=scheduler.patience,
                                                              factor=0.00001,
                                                              verbose=verbose)
        # add Scheduler
        elif scheduler.name == "CosineAnnealingWarmRestarts":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer,
                                                                                T_0=scheduler.t0,
                                                                                T_mult=scheduler.tmult,
                                                                                eta_min=scheduler.etaMin)
        else:
            return None
        return lr_scheduler

    def valid(self, model, criterion, val_loader):
        model.eval()
        val_bar = tqdm(val_loader)
        all_preds = []
        all_labels = []
        total_loss_valid = 0
        with torch.no_grad():
            for step, batch in enumerate(val_bar):
                x = {key: val.to(self.device) for key, val in batch[0].items()}
                y = batch[1].to(self.device)
                outputs = model(**x)  # attention mask 추가 
                loss_v = criterion(outputs.logits.squeeze(), y.squeeze()) # validation Loss라서 없어도 됨 <- # 성능 확인 시 필요할 수 있을 것 같아 우선 추가함
                total_loss_valid += loss_v.item()

                # Batch 별로 예측한 데이터와 label값들을 전체 데이터로 넣어줌
                all_preds.append(outputs.logits.squeeze())
                all_labels.append(y.squeeze())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        avg_loss_valid = total_loss_valid / len(val_loader)
        pearson = torchmetrics.functional.pearson_corrcoef(all_preds, all_labels) # Pearson 상관계수

        return avg_loss_valid, pearson
    
    
    def train(self, train_loader, val_loader):
        # Set initial
        model = self.get_model(self.model_name) 
        optim = self.get_optimizer(model=model, optimizer=self.optimizer)
        criterion = self.get_criterion(self.criterion)
        lr_scheduler = self.get_scheduler(optim, self.scheduler, verbose=True)
        model.to(self.device)
        current_best_loss = float('inf')
        
        # model train 
        model.train()
        early_stopping = EarlyStopping(patience=self.scheduler.patience, delta=0.001)
        for epoch in range(self.max_epoch):
            train_bar = tqdm(train_loader)
            total_loss_train = 0
            for step, batch in enumerate(train_bar):
                x = {key: val.to(self.device) for key, val in batch[0].items()}
                y = batch[1].to(self.device)
                
                # Calc Loss
                outputs = model(**x) 
                loss = criterion(outputs.logits.squeeze(), y.squeeze())
                total_loss_train += loss.item()
                
                # update weights and Scheduler
                loss.backward()
                optim.step()
                optim.zero_grad()

                train_bar.desc=f"Train Epoch[{epoch+1}/{self.max_epoch}] loss : {loss}"
                if lr_scheduler is not None:

                    if self.scheduler.name == "ReduceLROnPlateau":
                        lr_scheduler.step(loss) # Epoch이 너무 짧으므로 batch에 scheduler 도입

                    elif self.scheduler.name == "CosineAnnealingWarmRestarts":
                        lr_scheduler.step() # Epoch이 너무 짧으므로 batch에 scheduler 도입
            
            # 해당 epoch 내 평균 training loss
            avg_loss_train = total_loss_train / len(train_loader)

            # Epoch별 Validation
            avg_loss_valid, pearson = self.valid(model, criterion, val_loader)

            # epoch별 결과값 출력
            print("="*100)
            print(f"Training Loss (Average): {avg_loss_train:.4f} | Validation Loss (Average): {avg_loss_valid:.4f} | Pearson Coeff : {pearson:.4f}")
            print("="*100)

            # Early Stop 여부 확인
            if self.early_stop:
                early_stopping(avg_loss_valid)
                if early_stopping.early_stop:
                    print("Early stopping triggered")
                    break

            # validation loss에 따라 Ckpt 저장
            if avg_loss_valid < current_best_loss: # validation_loss 저장
                if self.use_wandb: # 갱신될 때만 logging
                    wandb.log({"validation_loss": avg_loss_valid, "validation_pearson": pearson, "epoch": epoch}) 
            
                prev_best_loss = current_best_loss
                current_best_loss = avg_loss_valid
                
                result = {
                    "model_name":self.model_name,
                    "optim":optim,
                    "epoch":epoch,
                    "prev_best_loss":prev_best_loss,
                    "current_best_loss":current_best_loss,
                    "pearson_valid":pearson
                }
                if self.hpo == "optuna":
                    # Change model_name to include the current trial number. 
                    result["model_name"] = f"{self.model_name}_trial_{self.trial.number}"

                ckpt_save(model, result)

            
        return avg_loss_valid, pearson
        
            
    def predict(self, model, dataloader):
        model.eval()
        all_preds = []
        with torch.no_grad():
            predict_bar = tqdm(dataloader)
            for step, batch in enumerate(predict_bar):
                x = {key: val.to(self.device) for key, val in batch.items()}
                predict = model(**x)
                
                all_preds.append(predict.logits.squeeze())
        
        predictions = torch.cat(all_preds)
        
        return predictions
