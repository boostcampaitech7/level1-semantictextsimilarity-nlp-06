import yaml
from box import Box
import torch
import os
import glob

def load_config(config_file):
    # Load Config.yaml
    with open(config_file) as file:
        config = yaml.safe_load(file) # Dictionary
        config = Box(config) # . 

    return config

def ckpt_save(model, model_name, optimizer, epoch, pearson, best_pearson):

    model_directory = "./saved_model"
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    
    # delete previous best-model
    previous_best = glob.glob(os.path.join(model_directory, f"{model_name.split('/')[1]}_best_model_Pearson_*_epoch_*.pt"))
    for file in previous_best:
        os.remove(file)

    # save new best-model
    save_path = f"{model_name.split('/')[1]}_best_model_Pearson_{pearson:.4f}_epoch_{epoch}.pt"
    torch.save(model, os.path.join(model_directory,save_path))

    print(f"Model improved from Pearson {best_pearson:.4f} to {pearson:.4f} and saved.")