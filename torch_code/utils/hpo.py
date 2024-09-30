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
    def __init__(self, n_trials, config):
        """
        Construct attributes for the HyperParameterOptimizer

        n_trials: int
            number of optuna trials
        config: object
            config for training model
        """
        self.n_trials = n_trials  # number of optuna trials
        self.sampler = TPESampler()  # sampling method for the hyperparameters
        self.pruner = HyperbandPruner()  # pruning method for the unpromising trials
        self.direction = "minimize"  # minimize or maximize the return of the optuna study
        self.config = config  # config for the training model
        self.trainer = None  # trainer for the training model
        self.dataloader = None  # dataloader for the model
        self.wandb_kwargs = {"project": "my-project"}  # wandb project setting

    def set_project_name(self, project_name="temp"):
        """
        Set the wandb project name.

        project_name: str
            name of the project
        """
        self.wandb_kwargs = {"project": project_name}

    def set_log(self, level=optuna.logging.INFO):
        """
        로그 레벨 설정

        level: int
            logging level
        """
        optuna.logging.set_verbosity(level)

    def init_config(self, trial):
        """
        Initialize the config from optuna trial config.

        trial: optuna.trial.Trial
            A trial is a process of evaluating an objective function.
        """
        self.config.training.batch_size = trial.suggest_int("batch_size", 32, 32)
        self.config.normalization = trial.suggest_categorical("normalization", [False])
        self.config.training.max_epoch = trial.suggest_int("max_epoch", 8, 8)
        self.config.training.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-4)
        self.config.training.scheduler.patience = trial.suggest_int("patience", 3, 3)
        self.config.training.scheduler.name = trial.suggest_categorical("scheduler_name", ["CosineAnnealingWarmRestarts"])
        self.config.training.scheduler.t0 = trial.suggest_int("t0", 2, 2)
        self.config.training.scheduler.tmult = trial.suggest_int("tmult", 2, 2)
        self.config.training.scheduler.etaMin = trial.suggest_float("etaMin", 1e-5, 1e-4)
        self.config.training.optimizer.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3)
        self.config.training.criterion = trial.suggest_categorical("criterion", ["MSELoss"])

    def set_sampler(self, sampler_name):
        """
        Set the sampler for the hyperparameters.

        sampler_name: str
            name of the sampler
        """
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
        """
        Set the pruner for optuna trials.

        pruner_name: str
            name of the pruner
        """
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
        """
        Target function for optuna hyperparameter optimiation.

        trial: optuna.trial.Trial
            A trial is a process of evaluating an objective function.
        """

        # Pivot is used by optuna to evaluate this trial.
        # If self.direction == "maximize" to maximize return value, initialize the pivot to -inf
        # else initialize the pivot to inf.
        pivot = float('inf')
        if self.direction == "maximize":
            pivot = -pivot
        
        try:
            # W&B run 초기화
            wandb.init(project=self.wandb_kwargs['project'])
        
            # Initialize the config with the suggested hyperparameters for the current trial.
            self.init_config(trial)
            config = self.config

            print("-----")
            print("The next trial hyperparameters")
            print(config)

            # Initialize the dataloader with the given config.
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
            # Setup the dataloader for training.
            dataloader.setup(stage="fit")

            # Store the initialized dataloader and trainer for further usage.
            self.dataloader = dataloader
            self.trainer = TorchTrainer(config, use_wandb=wandb.run)

            # Pass the trial object to the trainer, to log trial information.
            self.trainer.set_trial(trial)

            # Load the training and validation data.
            train_loader = dataloader.train_dataloader()
            val_loader = dataloader.val_dataloader()

            # Train the model by the trainer and obtain the loss and Pearson correlation from validation.
            loss, pearson = self.trainer.train(train_loader, val_loader)

            # Update the pivot to the validation loss.
            pivot = loss

            # Save the current hyperparameters.
            save_config(config, file_path=f"./saved_model/{config.model_name.split('/')[-1]}_trial_{trial.number}.yaml")

            # W&B에 메트릭 기록
            wandb.log({'final_loss': loss, 'final_pearson': pearson, **trial.params})
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed with error: {e}")
            print("##### Warning or Error #####")
            print("The current trial is stopped by an error.")
            print("If it is a GPU out of memory problem, reduce the batch size.")
        finally:
            # W&B run 종료
            wandb.finish()

        return pivot
    
    def optimize(self, direction="minimize"):
        """
        Optimize the hyperparameters using optuna.

        direction: str
            the direction of optimization, "minimize" or "maximize"
        """
        # Set up logging
        self.set_log()
        # wandb callback for logging
        wandbc = WeightsAndBiasesCallback(wandb_kwargs=self.wandb_kwargs)

        # Set the direction of optimization.
        self.direction = direction
        # Create an optuna study.
        self.study = optuna.create_study(direction=self.direction, sampler=self.sampler, pruner=self.pruner)
        # Start the optimization.
        self.study.optimize(self.objective, n_trials=self.n_trials, callbacks=[wandbc])

        df = self.study.trials_dataframe()
        # Print the n_trials results, sorted by value.
        print(df.sort_values(by='value'))