import torch
from torch.utils.data import Dataset
import pandas
from tqdm import tqdm
from transformers import AutoTokenizer


def preprocess(task, data_path, model_name): # task: "train", "valid", "test"
    data_df = pandas.read_csv(data_path)
    raw_data = data_df[['sentence_1', 'sentence_2']]

    tokenizer = AutoTokenizer.from_pretrained(model_name, max_length=160)

    labels = []
    if task=="train" or task=="valid":
        labels = data_df['label'].values.tolist()

    inputs = {
        "input_ids": [],
        "token_type_ids": [],
        "attention_mask": []
    }
    for _, sentence in tqdm(raw_data.iterrows(), desc='tokenization', total=len(raw_data)):
        sequence = '[SEP]'.join([sentence.iloc[0], sentence.iloc[1]])
        tokens = tokenizer(sequence, add_special_tokens=True, padding='max_length', truncation=True)
        inputs["input_ids"].append(tokens["input_ids"])
        inputs["token_type_ids"].append(tokens["token_type_ids"])
        inputs["attention_mask"].append(tokens["attention_mask"])
    
    dataset = STSDataset(inputs, labels)

    return dataset


class STSDataset(Dataset):
    def __init__(self, inputs, labels):
        super(STSDataset, self).__init__()
        self.inputs = inputs
        self.labels = labels
    
    def __len__(self):
        return len(self.inputs['input_ids'])
    
    def __getitem__(self, idx):
        if len(self.labels)==0: # test dataset에는 label이 주어지지 않음
            return {"input_ids": torch.tensor(self.inputs["input_ids"][idx]),
                    "token_type_ids": torch.tensor(self.inputs["token_type_ids"][idx]),
                    "attention_mask": torch.tensor(self.inputs["attention_mask"][idx])}
        
        return {"input_ids": torch.tensor(self.inputs["input_ids"][idx]),
                "token_type_ids": torch.tensor(self.inputs["token_type_ids"][idx]),
                "attention_mask": torch.tensor(self.inputs["attention_mask"][idx]),
                "labels": torch.tensor(self.labels[idx])}
