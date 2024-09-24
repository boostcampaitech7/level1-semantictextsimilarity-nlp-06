import optuna
from optuna.samplers import * 
from optuna.pruners import * 
from optuna.integration.wandb import WeightsAndBiasesCallback
import wandb

from dataset.dataloader import TextDataloader
from utils.Trainer import TorchTrainer
from utils.utils import ckpt_save, save_config

import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HyperParameterOptimizer:
    def __init__(self, n_trials, config):#, trainer, dataloader):
        self.n_trials = n_trials
        self.sampler = TPESampler()
        self.pruner = HyperbandPruner()
        self.direction = "minimize"
        self.config = config
        self.trainer = None #trainer
        self.dataloader = None #dataloader
        self.wandb_kwargs = {"project": "my-project"}

    def set_project_name(self, project_name="temp"):
        self.wandb_kwargs = {"project": project_name}

    def set_log(self, level=optuna.logging.INFO):
        # 로그 레벨 설정
        optuna.logging.set_verbosity(level)

    def init_config(self, trial):
        
        self.config.training.batch_size = trial.suggest_int("batch_size", 16, 16) # 2, 28
        self.config.normalization = trial.suggest_categorical("normalization", [False])
        self.config.training.max_epoch = trial.suggest_int("max_epoch", 5, 5) # 1, 10
        self.config.training.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-4)
        #self.config.training.scheduler.patience = trial.suggest_int("patience", 1, 2) # 10% of max_epoch is not bad
        self.config.training.scheduler.name = trial.suggest_categorical("scheduler_name", ["ReduceLROnPlateau", "CosineAnnealingWarmRestarts"])
        self.config.training.scheduler.t0 = trial.suggest_int("t0", 2, 2) # 5, 10
        self.config.training.scheduler.tmult = trial.suggest_int("tmult", 2, 2)
        self.config.training.scheduler.etaMin = trial.suggest_float("etaMin", 1e-6, 1e-5)
        #self.config.training.optimizer.name = trial.suggest_categorical("optimizer", ["AdamW"])
        self.config.training.optimizer.weight_decay = trial.suggest_float("weight_decay", 1e-4,1e-3)
        """
        MSELoss (Mean Squared Error Loss):
        보통 0.0001에서 0.001 사이의 값을 사용합니다1.
        L1Loss (Mean Absolute Error Loss):
        일반적으로 0.001에서 0.01 사이의 값을 사용합니다1.
        HuberLoss:
        Huber 손실 함수의 경우, 0.0001에서 0.01 사이의 값을 사용합니다1
        """
        self.config.training.criterion = trial.suggest_categorical("criterion", ["MSELoss", "L1Loss", "HuberLoss"])

        """
        trainer = self.trainer
        
        self.dataloader.batch_size = trial.suggest_int("batch_size", 2, 28)
        #trainer.batch_size = trial.suggest_int("batch_size", 2, 32)
        trainer.max_epoch = trial.suggest_int("max_epoch", 1, 5) # 30
        trainer.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3)
        trainer.scheduler.patience = trial.suggest_int("patience", 3, 10)
        #trainer.optimizer = trial.suggest_categorical("optimizer", ["AdamW"])
        trainer.optimizer.weight_decay = trial.suggest_float("weight_decay", 1e-6,1e-2)
        trainer.criterion = trial.suggest_categorical("criterion", ["MSELoss", "L1Loss", "HuberLoss"])

        #config.training.metric = trial.suggest_categorical("metric", ["loss", "pearson"])
        """
    
    #def set_config(self):
    #    pass

    def set_sampler(self, sampler_name):
        match sampler_name:
            case "Random":
                self.sampler = RandomSampler()
            case "Grid":
                self.sampler = GridSampler()
            case "TPE":
                self.sampler = TPESampler()
            case "CmaEs":
                self.sampler = CmaEsSampler()
            case "NSGAII":
                self.sampler = NSGAIISampler()
            case "NSGAIII":
                self.sampler = NSGAIIISampler()
            case "QMC":
                self.sampler = QMCSampler()
            case "GP":
                self.sampler = GPSampler()
            case "PartialFixed":
                self.sampler = PartialFixedSampler()
            case "BruteForce":
                self.sampler = BruteForceSampler()
    
    def set_pruner(self, pruner_name):
        match pruner_name:
            case "Base":
                self.pruner = BasePruner()
            case "Median":
                self.pruner = MedianPruner()
            case "Nop":
                self.pruner = NopPruner()
            case "Patient":
                self.pruner = PatientPruner()
            case "Percentile":
                self.pruner = PercentilePruner()
            case "SuccessiveHalving":
                self.pruner = SuccessiveHalvingPruner()
            case "Hyperband":
                self.pruner = HyperbandPruner()
            case "Threshold":
                self.pruner = ThresholdPruner()
            case "Wilcoxon":
                self.pruner = WilcoxonPruner()

    def objective(self, trial):

        pivot = float('inf')
        if self.direction == "maximize":
            pivot = -pivot
        
        try:
            # W&B run 초기화
            wandb.init(project=self.wandb_kwargs['project'])
        

            self.init_config(trial)
            config = self.config

            print("-----")
            print("The next trial hyperparameters")
            #print(trial.params)
            print(config)

            dataloader = TextDataloader(
                model_name=config.model_name,
                batch_size=config.training.batch_size,
                shuffle=config.training.shuffle,
                normalization = config.normalization,
                train_path=config.training.train_path,
                dev_path=config.test.dev_path,
                test_path=config.test.test_path,
                predict_path=config.test.predict_path
            )
            dataloader.setup(stage="fit")

            self.dataloader = dataloader
            self.trainer = TorchTrainer(config, use_wandb=wandb.run)
            self.trainer.set_trial(trial)

            train_loader = dataloader.train_dataloader()
            val_loader = dataloader.val_dataloader()

            loss, pearson = self.trainer.train(train_loader, val_loader)
            pivot = loss

            save_config(config, file_path=f"./saved_model/{config.model_name.split('/')[-1]}_{trial.number}.yaml")

            # W&B에 메트릭 기록
            wandb.log({'final_loss': loss, 'final_pearson': pearson, **trial.params})
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed with error: {e}")
            print("##### Warning or Error #####")
            print("The current trial is stopped by error.")
            print("If it is the batch size problem, the large batch size makes gpu error ")
            print("A new trial maybe start\n")
        finally:
            # W&B run 종료
            wandb.finish()

        return pivot
    
    def optimize(self, direction="minimize"):
        self.set_log()
        wandbc = WeightsAndBiasesCallback(wandb_kwargs=self.wandb_kwargs)

        self.direction = direction
        self.study = optuna.create_study(direction=self.direction, sampler=self.sampler, pruner=self.pruner)
        self.study.optimize(self.objective, n_trials=self.n_trials, callbacks=[wandbc])

        df = self.study.trials_dataframe()
        print(df.sort_values(by='value'))