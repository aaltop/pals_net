import os
import torch
import logging
import torch
import numpy as np

def convert_to_tensor(val):
    '''
    The preferred way to convert to PyTorch tensor due to certain problems
    with dtype.
    '''
    return torch.tensor(np.array(val), dtype=torch.float32)

def r2_score(pred,y):
    '''
    R-squared scoring for pytorch tensors.
    '''

    return (1-
            ((y-pred)**2).sum()/
            ((y-y.mean(dim=0))**2).sum()
        )

def save_model_state_dict(state_dict:dict, date_and_time:str, folder:str=None):
    '''
    Save a <state_dict>, such as the one returned from a PyTorch model's
    .state_dict() method. <folder> specifies the folder to save
    the data in, and defaults to "saved_models" if None.

    The file will be called model<date_and_time>.pt.
    '''

    if folder is None:
        folder = "saved_models"

    models_folder = os.path.join(
        os.getcwd(),
        folder
    )

    if not os.path.exists(models_folder):
        os.mkdir(models_folder)
    
    model_file_name = "model" + date_and_time + ".pt"
    file_path = os.path.join(models_folder,model_file_name)

    with open(file_path, "wb") as f:
        torch.save(state_dict,f)

    logging.info(f"Saved model to {file_path}")