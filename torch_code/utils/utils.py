import yaml
from box import Box
import torch
import os

def load_config(config_file):
    # Load Config.yaml
    with open(config_file) as file:
        config = yaml.safe_load(file)
        config = Box(config)

    return config

def ckpt_save(model, model_name, optimizer, epoch, pearson, best_pearson):

    model_directory = "./saved_model"
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    
    save_path = f"{model_name.split('/')[1]}_best_model_Pearson_{best_pearson}_epoch_{epoch}.pth"
    torch.save({
        'epoch':epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, os.path.join(model_directory,save_path))
    
    print(f"Model Saved at Pearson {best_pearson} to {pearson}")