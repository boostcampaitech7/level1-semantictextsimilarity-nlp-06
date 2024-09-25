from pandas import read_csv
from tqdm import tqdm
import numpy as np
import os

import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
from transformers import AutoTokenizer

from dataset import preprocess
from configurer import configurer


class Ensemble():
    def __init__(self, config_paths, train_path, valid_path, test_path):
        self.config_paths = config_paths
        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path
        
        self.y_train = read_csv(self.train_path)['label'].values.reshape(-1, 1)
        self.y_valid = read_csv(self.valid_path)['label'].values.reshape(-1, 1)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.X_train_base = self.base_prediction(self.train_path, self.config_paths, batch_size=16)
        self.X_valid_base = self.base_prediction(self.valid_path, self.config_paths, batch_size=16)
        self.X_test_base = self.base_prediction(self.test_path, self.config_paths, batch_size=1)

    def base_prediction(self, data_path, config_paths, batch_size):
        base_preds = None

        for config_path in config_paths:
            config = configurer(config_path)

            print(f"Right now using \"{config.model_name}\"")

            X_data = preprocess("test", data_path, config.model_name, False)
            X_loader = DataLoader(X_data, batch_size=batch_size, shuffle=False)

            pt_name = config.model_name.split('/')[1] + '.pt'
            model = torch.load(os.path.join('./models', pt_name))
            model.to(self.device)

            predictions = None
            for batch in tqdm(X_loader, "base prediction", total=len(X_loader)):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with torch.no_grad():
                    batch_prediction = model(**batch).logits.cpu().numpy()
                    if predictions is None:
                        predictions = batch_prediction
                    else:
                        predictions = np.concatenate([predictions, batch_prediction])
            
            if base_preds is None:
                base_preds = predictions
            else:
                base_preds = np.concatenate([base_preds, predictions], axis=1)
            
        print("\n")
        return base_preds

    def stacking(self, clf):
        self.classifier = None
        if clf=='linear':
            self.classifier = LinearRegression()
        elif clf=='lgbm':
            self.classifier = LGBMRegressor()

        self.classifier.fit(self.X_train_base, self.y_train)

        train_pred = self.classifier.predict(self.X_train_base)
        valid_pred = self.classifier.predict(self.X_valid_base)
        
        train_pearson, _ = pearsonr(train_pred.reshape(-1, 1), self.y_train)
        valid_pearson, _ = pearsonr(valid_pred.reshape(-1, 1), self.y_valid)

        print(f"============== {clf} result ==============")
        print(f"    train pearson sim: {train_pearson}")
        print(f"    valid pearson sim: {valid_pearson}")
        print(f"==========================================\n\n")
    
    def kfold_stacking(self, clf, n):
        kfold = KFold(n_splits=5)

        self.classifier = None
        if clf=="linear":
            self.classifier = LinearRegression()
        elif clf=="lgbm":
            self.classifier = LGBMRegressor()

        for train_idx, valid_idx in kfold.split(self.X_train_base):
            x_t, x_v = self.X_train_base[train_idx], self.X_train_base[valid_idx]
            y_t, y_v = self.y_train[train_idx], self.y_train[valid_idx]

            self.classifier.fit(x_t, y_t)

            train_pred = self.classifier.predict(x_t)
            k_valid_pred = self.classifier.predict(x_v)
            valid_pred = self.classifier.predict(self.X_valid_base)

            train_pearson, _ = pearsonr(y_t, train_pred.reshape(-1, 1))
            k_valid_pred, _ = pearsonr(y_v, k_valid_pred.reshape(-1, 1))
            valid_pred, _ = pearsonr(self.y_valid, valid_pred.reshape(-1, 1))

            print(f"=========== soft voting result ===========")
            print(f"    train pearson sim: {train_pearson}")
            print(f"    k-valid pearson sim: {k_valid_pred}")
            print(f"    valid pearson sim: {k_valid_pred}")
            print(f"==========================================\n\n")

    def soft_voting(self):
        train_pred = np.mean(self.X_train_base, axis=1)
        valid_pred = np.mean(self.X_valid_base, axis=1)

        train_pearson, _ = pearsonr(train_pred.reshape(-1, 1), self.y_train)
        valid_pearson, _ = pearsonr(valid_pred.reshape(-1, 1), self.y_valid)

        print(f"=========== soft voting result ===========")
        print(f"    train pearson sim: {train_pearson}")
        print(f"    valid pearson sim: {valid_pearson}")
        print(f"==========================================\n\n")
    
    def inference(self, is_voting, submission_path):
        submission = read_csv(submission_path)

        if is_voting:
            test_predictions = np.mean(self.X_test_base, axis=1)
        else:
            test_predictions = self.classifier.predict(self.X_test_base)
        
        submission['target'] = np.round(test_predictions, 1)
        submission.to_csv('ensemble_output.csv')

    def simulate(self, sentence_1, sentence_2, is_voting):
        sequence = '[SEP]'.join([sentence_1, sentence_2])

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        base_prediction = None
        for config_path in self.config_paths:
            config = configurer(config_path)

            print(f"Now Using {config.model_name}")
            
            pt_name = config.model_name.split('/')[1] + '.pt'
            tokenizer = AutoTokenizer.from_pretrained(config.model_name, max_length=160)
            model = torch.load(os.path.join('./models', pt_name))

            tokens = tokenizer(sequence, return_tensors='pt', add_special_tokens=True, padding='max_length', truncation=True)

            model.to(device)
            tokens.to(device)

            model.eval()
            with torch.no_grad():
                pred = model(**tokens).logits.cpu().numpy()
            
            if base_prediction is None:
                base_prediction = pred
            else:
                base_prediction = np.concatenate((base_prediction, pred), axis=1)
        
        label = None
        if is_voting:
            label = np.mean(base_prediction, axis=1)
        else:
            label = self.classifier.predict(base_prediction)
        label = np.round(label, 1)
        return label