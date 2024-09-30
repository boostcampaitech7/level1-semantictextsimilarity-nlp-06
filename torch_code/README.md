# BaseLine-for-PyTorch Code

PyTorch를 사용하여 한국어 STS를 수행할 수 있는 베이스라인 코드입니다.

## How To

`Base_config.yaml`

```
model_name: klue/roberta-small
normalization: True
training:
  batch_size: 32
  max_epoch: 1
  shuffle: True  
  early_stop: True
  learning_rate: 0.00002
  train_path': '../../data/train.csv'
  criterion: MSELoss
  optimizer:
    name: AdamW
    weight_decay: 0.01
  scheduler:
    name: CosineAnnealingWarmRestarts
    patience: 3
    t0: 5
    tmult: 2
    etaMin: 0.00001
  hpo: "" # "wandb_sweep" or "optuna"
test:
  dev_path': '../../data/dev.csv'
  test_path': '../../data/dev.csv'
  predict_path': '../../data/test.csv'
```



1. HuggingFace에 배포되어있는 모델과 토크나이저를 불러오기 위해 `model_name`을 설정합니다.
2. Normalization은 Label의 값 0~5를 0~1로 Scaling하는 설정값입니다.
3. Criterion은 `MSELoss`,`L1Loss`,`HuberLoss`로 지정되어있습니다.
4. Optimizer는 현재 `AdamW`만 구현되어있습니다.
5. Scheulder는 `CosineAnnealingWarmRestarts`와 `ReduceLROnPlateau` 중 하나를 사용 가능합니다.
   * `CosineAnnealingWarmRestarts` 사용시 t0, tmult, etamin 변수는 꼭 설정해주어야 합니다.
   * Scheduler는 Batch단위로 시행됩니다 (Epoch이 적게 실행되므로)
6. hpo는 하이퍼 파라미터 튜닝을 `wandb_sweep`과 `optuna`중 선정하여 사용합니다. 빈칸 일시 하이퍼파라미터튜닝을 진행하지 않습니다.



### How to train

```
python train.py --config Base_config --wandb
```

* `--config`는 .yaml을 제외한 config 파일 이름으로 설정
* `--wandb` 사용시 wandb로 모델 학습 과정을 visualization 가능
  * train.py내에 `os.environ["WANDB_API_KEY"] = ""  # "(본인 키 입력)" ` 를 설정해주어야함

train후에는 최종적으로 `{model_name}_loss_pearson_epoch.pt` 파일로 저장됩니다.



### How to inference

```
python inference.py --config Base_config --saved_model {model_name}_loss_pearson_epoch
```

* `--config`는 train과 동일하게 .yaml을 제외한 config 파일 이름으로 설정
* `--saved_model`은 train단계에서 저장된 .pt파일을 불러옴
  * 다른 모델을 불러올 시 config내에 `model_name`을 사용했던 Tokenizer로 맞춰주어야 tokenizer도 같이 불러올 수 있음



최종적으로 output/output.csv로 test dataset에 대한 predictions이 저장됩니다.









