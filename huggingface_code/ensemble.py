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

    def base_predictions(self):
        self.X_train_base = None
        self.X_valid_base = None
        self.X_test_base = None

        for config_path in self.config_paths:
            config = configurer(config_path)

            print(f"Right now using \"{config.model_name}\"")

            X_train = preprocess("test", self.train_path, config.model_name, False)
            X_valid = preprocess("test", self.valid_path, config.model_name, False)
            X_test = preprocess("test", self.test_path, config.model_name, False)

            X_train_loader = DataLoader(X_train, batch_size=16, shuffle=False)
            X_valid_loader = DataLoader(X_valid, batch_size=16, shuffle=False)
            X_test_loader = DataLoader(X_test, shuffle=False)

            # model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1)
            pt_name = config.model_name.split('/')[1] + '.pt'
            model = torch.load(os.path.join('./models', pt_name))
            model.to(self.device)

            train_predictions = None
            valid_predictions = None
            test_predictions = None
            for batch in tqdm(X_train_loader, "base prediction for train data", total=len(X_train_loader)):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with torch.no_grad():
                    batch_prediction = model(**batch).logits.cpu().numpy()
                    if train_predictions is None:
                        train_predictions = batch_prediction
                    else:
                        train_predictions = np.concatenate([train_predictions, batch_prediction])
            
            for batch in tqdm(X_valid_loader, "base prediction for valid data", total=len(X_valid_loader)):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with torch.no_grad():
                    batch_prediction = model(**batch).logits.cpu().numpy()
                    if valid_predictions is None:
                        valid_predictions = batch_prediction
                    else:
                        valid_predictions = np.concatenate([valid_predictions, batch_prediction])
            
            for batch in tqdm(X_test_loader, "base prediction for test data", total=len(X_test_loader)):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with torch.no_grad():
                    batch_prediction = model(**batch).logits.cpu().numpy()
                    if test_predictions is None:
                        test_predictions = batch_prediction
                    else:
                        test_predictions = np.concatenate([test_predictions, batch_prediction])

            if self.X_train_base is None:
                self.X_train_base = train_predictions
                self.X_valid_base = valid_predictions
                self.X_test_base = test_predictions
            else:
                self.X_train_base = np.concatenate([self.X_train_base, train_predictions], axis=1)
                self.X_valid_base = np.concatenate([self.X_valid_base, valid_predictions], axis=1)
                self.X_test_base = np.concatenate([self.X_test_base, test_predictions], axis=1)

            print("\n")

        return self.X_train_base, self.y_train, self.X_valid_base, self.y_valid, self.X_test_base
    
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