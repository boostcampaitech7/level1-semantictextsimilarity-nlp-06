# BaseLine-for-HuggingFace
Hugging Face의 Trainer를 사용하여 한국어 STS를 수행할 수 있는 베이스라인 코드입니다.

## How To

config.yaml
```yaml
sweeps: True
model: klue/roberta-large
lora: True
train_path: ./train.csv
valid_path: ./dev.csv
scale: True
output_dir: ./result/
epoch: 10
batch_size: 16
lr_init: 5e-5
lr_scheduler: cosine
weight_decay: 0.001
```  
**sweeps**  
- True로 지정 시 wandb의 sweeps 기능을 사용해서 하이퍼파라미터 튜닝을 시작  
- sweeps config는 model.py에서 설정할 수 있습니다. 또한 sweeps를 사용할 때는 config.yaml에서 epoch, batch_size, lr_init, weight_decay를 설정하지 않아도 됨  

**model**  
- Hugging Face에서 불러올 수 있는 uid를 지정하기  
- 현재 `AutoModelForSequenceClassification`을 사용하고 있기 때문에 사용하고자 하는 모델 카드나 README.md 참고  

**lora**  
- PEFT 기법 중 LoRA를 적용할 것인지, 말 건지를 결정  
- [model.py](./model.py)에서 LoraConfig를 지정  

**output_dir**  
- checkpoint를 저장할 dir 이름 지정  
- 지정된 이름의 dir이 없으면 새로 생성, 있으면 그대로 덮어씌우기  

**optimizer**  
- AdamW로 자동 설정  

**lr_scheduler**  
- `linear`, `constant`, `cosine` 중에 선택  

### How to train
```bash
python train.py --config ./sample_config.yaml
```

### How to inference
```bash
python test.py --is_peft {whehter the model tuned by peft} --data_path {path to test.csv} --is_scaled {whether the label normalization applied} --model_path {path to model checkpoint dir} --submit_path {path to submission.csv}
```  

### LoRA & Ensemble
LoRA를 적용한 튜닝은 [`peft_test.ipynb`](./peft_test.ipynb)에서 확인할 수 있습니다.  
학습이 완료된 모델을 사용한 앙상블은 [`ensemble.ipynb`](./ensemble.ipynb)에 들어가서 쉘을 하나씩 실행해보세요.  
(참고: 앙상블은 [torch_code](../torch_code/)에서 튜닝한 모델과 config를 기준으로 구현)

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

### train.py [click](./train.py)
`config.yaml`에서 제공한 정보대로 모델을 생성합니다.  
초기 learning rate, learning rate scheduler, weight decay는 모두 지정할 수 있지만, optimizer의 경우 일반적으로 가장 많이 사용되는 AdamW를 사용합니다.  
config에서 지정한 `output_dir`에 checkpoint 폴더가 생성되며 모델의 메타 데이터가 저장됩니다.  
모델 코드는 `model.py`에서 확인할 수 있습니다.  

### test.py [click](./test.py)
모델의 메타 데이터가 저장된 dir의 이름을 argument로 주면 학습 완료 후 저장했던 모델을 불러옵니다.  
label이 없는 test data에 대해 추론하고 submission.csv에 저장합니다.  
