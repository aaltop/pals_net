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


def pretty_print(obj, state_dict=None):
    '''
    PyTorch-style pretty printing, taken from the PyTorch Optimizer
    code with some modifications.

    Parameters
    ----------

    ### obj
        (Pytorch) object with `state_dict()` method which returns
        a dictionary. Alternatively,
        <obj> can be any given instance of a class, if <state_dict>
        is passed.

    ### state_dict : dict, default None.
        Any given dictionary. If not None, will be used instead of the state_dict
        returned by <obj>'s `state_dict()`.

    '''

    format_string = type(obj).__name__ + ' ('
    format_string += '\n'

    if state_dict is None:
        state_dict = obj.state_dict()

    for key,value in sorted(state_dict.items()):
        if key != 'params':
            format_string += f'    {key}: {value}\n'
    format_string += ')'

    return format_string


def conv_expand(tensor):
    '''
    Expand <tensor> of shape (batch_size, signal_length) to the 
    shape (batch_size, channel_count = 1, signal_length) or shape (signal_length) to the 
    shape (channel_count = 1, signal_length) for use
    with torch.nn.Conv1d.

    At least 1D convolutions in PyTorch require input to be in the shape
    (batch_size, channel_count, signal_length) or 
    (channel_count, signal_length), which is difficult
    to work with when input to a model is otherwise (batch_size, signal_length).
    '''

    length = len(tensor.shape)

    match length:
        case 2:
            batch_size, signal_length = tensor.shape
            return tensor.reshape((batch_size, 1, signal_length))
        case 1: # in case the tensor is just one batch as one dimension
            return tensor.reshape((1,-1))
        case _:
            return tensor


def conv_compress(tensor):
    '''
    Compress <tensor> of shape (batch_size, channel_count = 1, signal_length) to the 
    shape (batch_size, signal_length) or shape (channel_count = 1, signal_length) to the 
    shape (signal_length) for use with torch.nn.Conv1d.

    see also `conv_expand()`.
    '''

    length = len(tensor.shape)

    match length:
        case 3:
            batch_size, _, signal_length = tensor.shape
            return tensor.reshape((batch_size, signal_length))
        case 2:
            return tensor.reshape((tensor.shape[-1]))
        case _:
            raise ValueError("Shape of <tensor> should be 2 or 3.")



def test_pretty_print():

    optim = torch.optim.Adam([torch.tensor(1)])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim)

    print(pretty_print(sched))

if __name__ == "__main__":

    test_pretty_print()