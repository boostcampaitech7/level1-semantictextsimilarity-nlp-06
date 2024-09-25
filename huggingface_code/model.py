import numpy as np
from scipy.stats import pearsonr
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

import wandb
from peft import LoraConfig, TaskType, get_peft_model


class Model():
    def __init__(self, sweeps, lora, model_name, output_dir, epoch, train_data, valid_data, batch_size, lr, lr_scheduler, weight_decay):
        self.sweeps = sweeps
        
        self.output_dir = output_dir
        self.epoch = epoch
        self.train_data = train_data
        self.valid_data = valid_data
        self.batch_size = batch_size
        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.weight_decay = weight_decay
        
        self.model_name = model_name

        if self.sweeps:
            self.sweep_config = {
                "method": "bayes",
                "metric": {
                    "name": "pearson_corr",
                    "goal": "maximize"
                },
                "early_terminate": {
                    'type': 'hyperband',
                    's': 3,
                    'eta': 3,
                    'max_iter': 100
                },
                "parameters": {
                    'max_epoch': {'values': [1, 3, 5]},
                    'batch_size': {'value': 16},
                    'learning_rate': {
                        'distribution': 'uniform',
                        'min': 1e-5,
                        'max': 1e-3
                    },
                    'weight_decay': {
                        'distribution': 'uniform',
                        'min': 0.001,
                        'max': 0.1
                    },
                    'scheduler_patience': {'values': [3, 5, 7]}
                }
            }
            self.sweep_id = wandb.sweep(self.sweep_config, project=self.model_name.split('/')[1])

        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=1)
        for param in self.model.parameters(): param.data = param.data.contiguous()

        if lora:
            self.peft_config = LoraConfig(task_type=TaskType.SEQ_CLS,
                                          inference_mode=False,
                                          r=8,
                                          lora_alpha=16,
                                          lora_dropout=0.1)
            self.model = get_peft_model(self.model, self.peft_config)
        
        if self.sweeps:
            def train(config=None):
                with wandb.init(config=config):
                    config = wandb.config

                    self.training_arguments = TrainingArguments(
                        output_dir=self.output_dir,
                        report_to='wandb',
                        overwrite_output_dir=True,
                        num_train_epochs=config.max_epoch,
                        eval_strategy="epoch",
                        per_device_train_batch_size=config.batch_size,
                        per_device_eval_batch_size=config.batch_size,
                        lr_scheduler_type=self.lr_scheduler, # 'linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup', 'inverse_sqrt'
                    )

                    self.trainer = Trainer(
                        model=self.model,
                        args=self.training_arguments,
                        train_dataset=self.train_data,
                        eval_dataset=self.valid_data,
                        compute_metrics=self.compute_metrics,
                    )

                    self.trainer.train()
            wandb.agent(self.sweep_id, train, count=10)
        else:
            self.training_arguments = TrainingArguments(
                output_dir=self.output_dir,
                overwrite_output_dir=True,
                num_train_epochs=self.epoch,
                eval_strategy="epoch",
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=self.batch_size,
                weight_decay=self.weight_decay,
                learning_rate=self.lr,
                lr_scheduler_type=self.lr_scheduler, # 'linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup', 'inverse_sqrt'
            )

            self.trainer = Trainer(
                model=self.model,
                args=self.training_arguments,
                train_dataset=self.train_data,
                eval_dataset=self.valid_data,
                compute_metrics=self.compute_metrics,
            )

            self.trainer.train()

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        
        predictions = logits.squeeze()
        labels = np.array(labels)

        pearson_corr, _ = pearsonr(predictions, labels)

        return {"pearson_corr": pearson_corr}
