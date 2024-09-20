import yaml
from box import Box
import torch
import os

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
    
    save_path = f"{model_name.split('/')[1]}_best_model_Pearson_{pearson}_epoch_{epoch}.pt"
    torch.save(model, os.path.join(model_directory,save_path))
    
    print(f"Model Saved at Pearson {best_pearson} to {pearson}")