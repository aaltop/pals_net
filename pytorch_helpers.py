import os
import torch
import logging
import torch
import numpy as np

def convert_to_tensor(val, device=None, dtype=None):
    '''
    The preferred way to convert to PyTorch tensor due to certain problems
    with dtype.

    Parameters
    ----------

    ### val : an iterable of numbers
    
    ### device : string, default None
        torch.device to use. If None, defaults to "cpu".
    
    ### dtype : Pytorch dtype, default None
        Datatype to use. If None, defaults to torch.float32.

    '''

    if device is None:
        device = "cpu"
    
    if dtype is None:
        dtype = torch.float32

    return torch.tensor(np.array(val), dtype=dtype, device=device)

def r2_score(pred,y):
    '''
    R-squared scoring for pytorch tensors. Excepts <y> and <pred>
    to be column-oriented if 2D: the sums as well as the mean of <y> will be 
    taken column-wise.
    '''

    return (1-
            ((y-pred)**2).sum(dim=0)/
            ((y-y.mean(dim=0))**2).sum(dim=0)
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


def pretty_print(obj):
    '''
    PyTorch-style pretty printing, taken from the PyTorch Optimizer
    code with some modifications.
    '''

    format_string = obj.__class__.__name__ + ' ('
    format_string += '\n'

    for key,value in sorted(obj.state_dict().items()):
        if key != 'params':
            format_string += f'    {key}: {value}\n'
    format_string += ')'

    return format_string
    
def test_pretty_print():

    optim = torch.optim.Adam([torch.tensor(1)])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim)

    print(pretty_print(sched))

if __name__ == "__main__":

    test_pretty_print()