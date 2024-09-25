import argparse
from transformers import AutoModelForSequenceClassification
from peft import AutoPeftModelForSequenceClassification
from torch.utils.data import DataLoader
import torch
from pandas import read_csv
import os
import json

from dataset import preprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_peft", type=bool, default=True, help='whether the model is peft')
    parser.add_argument("--is_scaled", type=bool, default=True, help='whether the labels are scaled')
    parser.add_argument("--data_path", type=str, default='./test.csv', help="path to test.csv")
    parser.add_argument("--model_path", type=str, default='./results/checkpoint-584', help="dir path to safetensors")
    parser.add_argument("--submit_path", type=str, default='../../sample_submission.csv', help="path to sample_submission.csv")
    arg = parser.parse_args()

    if arg.is_peft == "True":
        with open(os.path.join(arg.model_path, 'adapter_config.json')) as f:
            model_config = json.load(f)

        model = AutoPeftModelForSequenceClassification.from_pretrained(arg.model_path, num_labels=1)

        test_data = preprocess(task="test", data_path='../../../data/test.csv', model_name=model_config['base_model_name_or_path'], scale=arg.is_scaled)
        test_loader = DataLoader(test_data, shuffle=False)
    else:
        with open(os.path.join(arg.model_path, 'config.json')) as f:
            model_config = json.load(f)
        
        model = AutoModelForSequenceClassification.from_pretrained(arg.model_path, num_labels=1)

        test_data = preprocess(task="test", data_path=arg.data_path, model_name=model_config['_name_or_path'], scale=arg.is_scaled)
        test_loader = DataLoader(test_data, shuffle=False)

    device = 'cuda' if torch.cuda.is_available else 'cpu'
    model.to(device)

    outputs = []
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            predictions = model(**batch)
            outputs.append(predictions.logits)
    
    if arg.is_scaled:
        outputs = list(round(float(i*5.), 1) for i in outputs)
    else:
        outputs = list(round(float(i), 1) for i in outputs)

    sample_csv = read_csv(arg.submit_path)
    sample_csv['target'] = outputs

    sample_csv.to_csv(os.path.join(arg.model_path, 'output.csv'), index=False)
