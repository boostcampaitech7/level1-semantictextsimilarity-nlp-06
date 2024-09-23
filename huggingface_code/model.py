import numpy as np
from scipy.stats import pearsonr
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

from peft import LoraConfig, TaskType, get_peft_model


class Model():
    def __init__(self, lora, model_name, output_dir, epoch, train_data, valid_data, batch_size, lr, lr_scheduler, weight_decay):
        self.model_name = model_name

        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=1)

        if lora:
            self.peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1)
            self.model = get_peft_model(self.model, self.peft_config)
    
        self.training_arguments = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=epoch,
            learning_rate=lr,
            weight_decay=weight_decay,
            eval_strategy="epoch",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            lr_scheduler_type=lr_scheduler, # 'linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup', 'inverse_sqrt'
        )

        self.trainer = Trainer(
            model=self.model,
            args=self.training_arguments,
            train_dataset=train_data,
            eval_dataset=valid_data,
            compute_metrics=self.compute_metrics,
        )

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        
        predictions = logits.squeeze()
        labels = np.array(labels)

        pearson_corr, _ = pearsonr(predictions, labels)

        return {"pearson_corr": pearson_corr}

    def train(self):
        self.trainer.train()
