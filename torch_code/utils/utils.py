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

def save_config(config, file_path="./saved_model/config.txt"):
    # save config.yaml
    with open(file_path, 'w') as file:    
        yaml.dump(config, file)

def ckpt_save(model, result): #model_name, optimizer, epoch, loss, minimum_loss):
    # result : Dict()
    
    model_name = result["model_name"]
    optim = result["optim"]
    epoch = result["epoch"]
    prev_loss = result["prev_best_loss"]
    current_loss = result["current_best_loss"]
    pearson_valid = result["pearson_valid"]

    model_directory = "./saved_model"
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    
    # delete previous best-model
    previous_best = glob.glob(os.path.join(model_directory, f"{model_name.split('/')[1]}_loss_*_pearson_*_epoch_*.pt"))
    for file in previous_best:
        os.remove(file)

    # save new best-model
    save_path = f"{model_name.split('/')[1]}_loss_{current_loss:.4f}_pearson_{pearson_valid:.4f}_epoch_{epoch}.pt"
    torch.save(model, os.path.join(model_directory,save_path))

    print(f"Model improved from validation loss {prev_loss:.4f} to {current_loss:.4f} and saved.")