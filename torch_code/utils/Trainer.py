from utils.utils import ckpt_save
import pandas as pd

from tqdm.auto import tqdm

import transformers
import torch
import torchmetrics
import pytorch_lightning as pl

class torch_Trainer():
    def __init__(self, config):

        self.model_name = config.model_name
        self.lr = config.training.learning_rate
        self.loss = config.training.loss
        self.batch_size = config.training.batch_size
        self.scheduler = config.training.scheduler
        self.optimizer = config.training.optimization
        self.epoch = config.training.max_epoch
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    def get_model(self, model_name):
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, num_labels=1
        )
        return model
        # if you want change custom model, you redefine this function

    def get_optimizer(self, model, optimizer):
        if optimizer.name == "AdamW":
            optim = torch.optim.AdamW(model.parameters(), weight_decay=optimizer.weight_decay,
                                      lr=self.lr)
        # Add Optim

        return optim

    def get_loss(self, loss):
        if loss == "MSELoss":
            return torch.nn.MSELoss()
        elif loss == "l1Loss":
            return torch.nn.L1Loss()
        elif loss == "HuberLoss":
            return torch.nn.HuberLoss()
        # Add Loss
    
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
        total_loss = 0
        with torch.no_grad():
            for step, batch in enumerate(val_bar):
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                outputs = model(x) # output) outputs.logits (예측결과)
                loss_v = criterion(outputs.logits.squeeze(), y.squeeze()) # validation Loss라서 없어도 됨 <- # 성능 확인 시 필요할 수 있을 것 같아 우선 추가함
                total_loss += loss_v.item()

                # Batch 별로 예측한 데이터와 label값들을 전체 데이터로 넣어줌
                all_preds.append(outputs.logits.squeeze())
                all_labels.append(y.squeeze())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        avg_loss = total_loss / len(val_loader)
        pearson = torchmetrics.functional.pearson_corrcoef(all_preds, all_labels) # Pearson 상관계수
        print("======================================================")
        print(f"        Validation Loss : {avg_loss:.4f}")
        print(f"        Pearson Coeff : {pearson:.4f}")
        print("======================================================")
        return pearson
    
    
    def train(self, train_loader, val_loader):
        # Set initial
        model = self.get_model(self.model_name) 
        optim = self.get_optimizer(model=model, optimizer=self.optimizer)
        criterion = self.get_loss(self.loss)
        lr_scheduler = self.get_scheduler(optim, self.scheduler, verbose=True)
        model.to(self.device)
        best_pearson = 0.0
        
        # model train 
        model.train()
        for epoch in range(self.epoch):
            train_bar = tqdm(train_loader)
            for step, batch in enumerate(train_bar):
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                
                # Calc Loss
                outputs = model(x) 
                loss = criterion(outputs.logits.squeeze(), y.squeeze())
                
                # update weights and Scheduler
                loss.backward()
                optim.step()
                optim.zero_grad()
                train_bar.desc=f"Train Epoch[{epoch+1}/{self.epoch}] loss : {loss}"
                if lr_scheduler is not None:
                    lr_scheduler.step() # Epoch이 너무 짧으므로 batch에 scheduler 도입
            # Epoch별 Validation
            pearson = self.valid(model, criterion, val_loader)
        
            # validation Pearson에 따라 Ckpt 저장
            if pearson > best_pearson: # Best Pearson 저장
                ckpt_save(model, self.model_name, optim, self.epoch, pearson, best_pearson)
                best_pearson = pearson
        
            
    def predict(self, model, dataloader):
        model.eval()
        all_preds = []
        with torch.no_grad():
            predict_bar = tqdm(dataloader)
            for step, batch in enumerate(predict_bar):
                x = batch
                x = x.to(self.device)
                predict = model(x)
                
                all_preds.append(predict.logits.squeeze())
        
        predictions = torch.cat(all_preds)
        
        return predictions
