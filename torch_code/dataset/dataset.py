import torch

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, token_type_ids, attention_masks, targets=[]):
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_masks = attention_masks
        self.targets = targets

    # Fetches one data point at a time during training and inference
    def __getitem__(self, idx):
        inputs = {
            'input_ids': torch.tensor(self.input_ids[idx]),
            'token_type_ids': torch.tensor(self.token_type_ids[idx]),
            'attention_mask': torch.tensor(self.attention_masks[idx])
        }
        # 정답이 있다면 else문을, 없다면 if문을 수행합니다
        if len(self.targets) == 0:
            return inputs
        else:
            return inputs, torch.tensor(self.targets[idx])

    # 입력하는 개수만큼 데이터를 사용합니다
    def __len__(self):
        return len(self.input_ids)