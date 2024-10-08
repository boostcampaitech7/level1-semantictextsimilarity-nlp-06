import torch
import transformers
from dataset.dataset import CustomDataset
from tqdm.auto import tqdm
import pandas as pd


class TextDataloader():
    def __init__(self, model_name, batch_size, shuffle, normalization, train_path, dev_path, test_path, predict_path):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.normalization = normalization

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, max_length=160)
        self.target_columns = ['label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']

    def tokenizing(self, dataframe):
        data = {
            "input_ids": [],
            "token_type_ids": [],
            "attention_mask": []
        }
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.
            text = '[SEP]'.join([item[text_column] for text_column in self.text_columns])
            outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True)
            data["input_ids"].append(outputs["input_ids"])
            data["token_type_ids"].append(outputs["token_type_ids"])
            data["attention_mask"].append(outputs["attention_mask"])
        return data

    def preprocessing(self, data):
        # 안쓰는 컬럼을 삭제합니다.
        data = data.drop(columns=self.delete_columns)

        # 타겟 데이터가 없으면 빈 배열을 리턴합니다.
        try:
            targets = data[self.target_columns].values.tolist()
        except:
            targets = []
        
        if self.normalization:
            targets = [[label[0]/5.0] for label in targets] # error 떠서 [[]] 구조로 바꿔봄
        
        # 텍스트 데이터를 전처리합니다.
        tokenized_data = self.tokenizing(data)

        inputs = tokenized_data["input_ids"]
        token_type_ids = tokenized_data["token_type_ids"]
        attention_masks = tokenized_data["attention_mask"]

        return inputs, token_type_ids, attention_masks, targets
    

    def setup(self, stage='fit'):
        if stage == 'fit':
            # 학습 데이터와 검증 데이터셋을 호출합니다
            train_data = pd.read_csv(self.train_path)
            val_data = pd.read_csv(self.dev_path)

            # 학습데이터 준비
            train_inputs, train_token_type_ids, train_attention_masks, train_targets = self.preprocessing(train_data)

            # 검증데이터 준비
            val_inputs, val_token_type_ids, val_attention_masks, val_targets = self.preprocessing(val_data)

            # train 데이터만 shuffle을 적용해줍니다, 필요하다면 val, test 데이터에도 shuffle을 적용할 수 있습니다
            self.train_dataset = CustomDataset(train_inputs, train_token_type_ids, train_attention_masks, train_targets)
            self.val_dataset = CustomDataset(val_inputs, val_token_type_ids, val_attention_masks, val_targets)
        else:
            # 평가데이터 준비
            test_data = pd.read_csv(self.test_path)
            test_inputs, test_token_type_ids, test_attention_masks, test_targets = self.preprocessing(test_data)
            self.test_dataset = CustomDataset(test_inputs, test_token_type_ids, test_attention_masks, test_targets)

            predict_data = pd.read_csv(self.predict_path)
            predict_inputs, predict_token_type_ids, predict_attention_masks, predict_targets = self.preprocessing(predict_data)
            self.predict_dataset = CustomDataset(predict_inputs, predict_token_type_ids, predict_attention_masks, [])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)