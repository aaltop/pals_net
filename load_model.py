import os

import torch

def load_train_dict(model_folder=None, model_file=None):
    '''
    Load the data saved during model training
    '''

    if model_folder is None:
        model_folder = "saved_models"

    model_path = os.path.join(os.getcwd(), model_folder)

    if model_file is None:
        model_file = os.listdir(model_path)[-1]

    path_to_saved_model = os.path.join(model_path, model_file)

    with open(path_to_saved_model, "rb") as f:
        train_dict = torch.load(f)

    return train_dict


def load_model(train_dict:dict=None, network:torch.nn.Module=None):
    '''
    Inititialise the model based on data in the <train_dict>.

    See `load_train_dict()` for more about <train_dict>.


    Parameters
    ----------

    ### train_dict
        Contains relevant values used in training. If None,
        uses `load_train_dict()` to get.

    ### network
        The class used as the model. If None,
        attempts to get the value at key "model_class" from <train_dict>.
    '''
    
    if train_dict is None:
        train_dict = load_train_dict()

    if network is None:
        network = train_dict["model_class"]

    dev = train_dict.get("device", "cpu")
    dtype = train_dict.get("dtype", torch.float32)


    # NOTE: this assumes state_dict contains the model_kwargs,
    # model_state_dict and the normalisation originally used for
    # the output, as well as used device and data type
    if "model_layers" in train_dict: # for older train_dict contents (MLP)
        model = network(train_dict["model_layers"]).to(dev, dtype)
    else:
        model = network(**train_dict["model_kwargs"]).to(dev, dtype)

    model.load_state_dict(train_dict["model_state_dict"])

    return model


if __name__ == "__main__":

    network = load_model()
    print(network)