# BaseLine-for-HuggingFace
Hugging Face의 Trainer를 사용하여 한국어 STS를 수행할 수 있는 베이스라인 코드입니다.

## How To

config.yaml
```yaml
output_dir: ./result/
train_path: ./train.csv
valid_path: ./dev.csv
batch_size: 16
model: klue/roberta-small
epoch: 1
lr_init: 5e-5
lr_scheduler: linear
weight_decay: 0.0001
```  
**output_dir**은 checkpoint를 저장할 dir 이름을 지정해주면 됩니다. 없으면 새로 만들어주고, 있으면 그대로 덮어씌워지므로 다른 모델로 바꿀 경우 다른 경로를 지정해주시는 게 좋습니다.  
**model**은 Hugging Face에서 불러올 수 있는 uid를 지정해주시면 됩니다. 현재 `AutoModelForSequenceClassification`을 사용하고 있기 때문에 사용하고자 하는 모델 카드나 README.md를 잘 읽어보시고 지정해주세요.
**optimizer**는 AdamW로 자동 설정되어 있습니다.  
Hugging Face에서 제공하는 Adagrad같은 optimizer도 있으나 일반적으로 AdamW를 가장 많이 사용한다고 해서 일단 변화하지 않게 두었는데 혹시 필요하시면 수정해보겠습니다.  
**lr_scheduler**는 `linear`, `constant`, `cosine` 중에 선택하면 되고, 추가로 매개변수가 필요한 scheduler의 경우 추가하지 않도록 두었습니다.

### How to train
```bash
python train.py --config ./config.yaml
```

### How to inference
```bash
python test.py --data_path {path to test.csv} --model_path {path to model checkpoint dir}
```  

## Code Explanation
### dataset.py
train, valid 데이터의 경우 Hugging Face의 Trainer를 사용하기 위해 아래와 같은 형태의 dictionary를 반환합니다.
```python
{
    "input_ids": [],
    "token_type_ids": [],
    "attention_mask": [],
    "labels": []
}
```  
test 데이터의 경우 `labels`가 없이 위와 동일한 형태의 dictionary를 반환합니다.  

### train.py
`config.yaml`에서 제공한 정보대로 모델을 생성합니다.  
초기 learning rate, learning rate scheduler, weight decay는 모두 지정할 수 있지만, optimizer의 경우 일반적으로 가장 많이 사용되는 AdamW를 사용합니다.  
config에서 지정한 `output_dir`에 checkpoint 폴더가 생성되며 모델의 메타 데이터가 저장됩니다.  
모델 코드는 `model.py`에서 확인할 수 있습니다.  

### test.py
모델의 메타 데이터가 저장된 dir의 이름을 argument로 주면 학습 완료 후 저장했던 모델을 불러옵니다.  
label이 없는 test data에 대해 추론하고 submission.csv에 저장합니다.  
