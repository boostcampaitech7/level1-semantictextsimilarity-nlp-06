import argparse
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader
import torch
from pandas import read_csv
import os
import json

from dataset import preprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='./test.csv', help="path to test.csv")
    parser.add_argument("--model_path", type=str, default='./results/checkpoint-584', help="dir path to safetensors")
    parser.add_argument("--submit_path", type=str, default='../../sample_submission.csv', help="path to sample_submission.csv")
    arg = parser.parse_args()

    with open(os.path.join(arg.model_path), 'config.json') as f:
        model_config = json.load(f)
    
    model = AutoModelForSequenceClassification.from_pretrained(arg.model_path)

    test_data = preprocess(task="test", data_path=arg.data_path, model_name=model_config['_name_or_path'])
    test_loader = DataLoader(test_data, shuffle=False)

    device = 'cuda' if torch.cuda.is_available else 'cpu'
    model.to(device)

    outputs = []
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            predictions = model(**batch)
            outputs.append(predictions.logits)
    
    outputs = list(round(float(i), 1) for i in torch.cat(outputs))

    sample_csv = read_csv('../../sample_submission.csv')
    sample_csv['target'] = outputs

    sample_csv.to_csv(os.path.join(arg.model_path, 'output.csv'), index=False)