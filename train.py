'''

# Instructions:

Everything that's needed to do here (to begin with) is contained in the arguments
of the `main()` function at the bottom. Main things to possibly
change in the function:

- <data_folder>: The folder where the data simulated in 
"simulate_spectra.py" was saved.

- <train_size>, <test_size>: the preferred sizes of training and test
data. The sum of these should be at most equal to the number of data
files in <data_folder>, with both values greater than zero.

After these are verified to be okay, it should be okay to run the
file to train the model.

The first time the training is run (during a session), it might take
a while depending on the amount of data specified. As an example,
processing of of 7500 train data and 500 test data took 70 seconds
the first time, but second time took around 4 seconds.
'''


import torch
import numpy as np
import logging
#pylint: disable=logging-not-lazy, logging-fstring-interpolation
import time
import os
import sys

import matplotlib.pyplot as plt

import helpers

from read_data import (
    read_metadata, 
    get_train_or_test,
    get_components,
    get_data_files,
    get_simdata
)

from pytorch_helpers import (
    r2_score, 
    save_model_state_dict, 
    convert_to_tensor
)

from pytorchMLP import MLP

_rng = np.random.default_rng()

# It's kind of a stupid name, but don't no what
# else to call it
class Model:
    '''

    For the training and evaluation of a pytorch model.
    
    Parameters
    ----------

    ### model : PyTorch model

    ### optim : PyTorch optimiser

    ### loss_func : PyTorch loss function
    
    '''

    def __init__(self, model, optim, loss_func):

        self.inner_model = model
        self.optim = optim
        self.loss_func = loss_func


    
    def train(self, inputs, outputs, batch_size):
        '''
        Does one iteration of model training.
        '''

        epoch_loss = 0

        self.inner_model.train()

        # TODO: could make a proper dataloader thing for this
        num_batches = inputs.shape[0]//batch_size

        # do at least one iteration
        if num_batches == 0:
            num_batches == 1

        for i in range(num_batches):
            x = inputs[i*batch_size:(i+1)*batch_size,:]
            y = outputs[i*batch_size:(i+1)*batch_size,:]


            self.optim.zero_grad()

            pred = self.inner_model(x)

            # print(pred)

            loss = self.loss_func(pred, y)
            loss.backward()
            epoch_loss += loss.item()

            self.optim.step()

        return epoch_loss/num_batches
    
    def evaluate(self, inputs, outputs, batch_size):
        '''
        Verbose evaluation of model performance


        Returns
        -------

        ### epoch_loss : float
            the average loss of the batch evaluations
        '''

        with torch.no_grad():

            epoch_loss = 0

            self.inner_model.eval()

            # TODO: could make a proper dataloader thing for this
            num_batches = inputs.shape[0]//batch_size
            if num_batches == 0:
                num_batches = 1

            for i in range(num_batches):
                x = inputs[i*batch_size:(i+1)*batch_size,:]
                y = outputs[i*batch_size:(i+1)*batch_size,:]

                pred = self.inner_model(x)

                logging.info("prediction:")
                logging.info(pred)
                logging.info("true:")
                logging.info(y)
                logging.info("difference:")
                logging.info((pred-y).abs())

                loss = self.loss_func(pred, y)
                logging.info("loss:")
                logging.info(loss.item())
                logging.info("R2:")
                logging.info(r2_score(pred,y).mean().item())
                epoch_loss += loss.item()

            logging.info("mean loss:")
            logging.info(epoch_loss/num_batches)

        return epoch_loss
    
    def r2_evaluate(self, inputs, outputs, do_logging=False):
        '''
        Evaluates the R-squared score, optionally does logging to log the values.

        Returns
        -------

        ### r2
        '''

        self.inner_model.eval()
        with torch.no_grad():

            r2 = r2_score(self.inner_model(inputs), outputs).mean().item()

        if do_logging:
            logging.info("\nR-squared:")
            logging.info(r2)

        return r2
    

def process_input(data, num_of_channels=None, take_average_over=None, start_index=None):
    '''
    Cut off counts in <data> such that only data from the
    <start_index> onwards up to <num_of_channels> beyond the <start_index> 
    is considered, and takes non-rolling means of <take_average_over> values 
    of the result. Finally, normalises the resulting averaged data.

    Parameters
    ----------
    ### data : list of numpy arrays
        The counts for <data.size> simulated spectra.

    ### num_of_channels : int
        How many channels to take beyond the max channel,
        as in [max_chan:max_chan+num_of_channels]. Preferably
        divisible by <take_average_over>. Default is 100.

    ### take_average_over : int
        how many channels to average over. Default is 5.

    ### start_index : int
        The first index to include in the end result. If None,
        takes the index of the max value.

    
    Returns
    -------
        None, the operation is performed in-place.
    '''

    if num_of_channels is None:
        num_of_channels = 100
    if take_average_over is None:
        take_average_over = 5

    for i in range(len(data)):
        if start_index is None:
            start_index = data[i].argmax()
        # average over <take_average_over> value groups (non-rolling)
        used_data = data[i][start_index:start_index+num_of_channels]
        # take a divisible number of data points
        data_length = (used_data.flatten().shape[0]//take_average_over)*take_average_over
        
        averaged_data = used_data[:data_length].reshape((-1,take_average_over)).mean(axis=1)
        # normalise
        data[i] = averaged_data/averaged_data.max()

def fetch_and_process_input(
        data_folder, 
        data_files, 
        take_average_over=None,
        start_index=None
        ):
    '''
    Fetch and process train/test input data.

    Parameters
    ----------

    ### data_files : list of strings
        The names of the files which contain the input data. Should
        be a list also for just one file.


    ### take_average_over : int
        In processing the input, a non-rolling average is calculated.
        This determines how many data points are included in each 
        averaging.
        


    Returns
    -------

    ### inputs : PyTorch tensor
        The input data.


    '''

    folder_path = os.path.join(os.getcwd(), data_folder)

    inputs = get_train_or_test(folder_path, data_files)

    # not necessary sensible, but ensures that all channels beyond
    # start_index are taken (could of course be changed)
    num_of_channels = inputs[0].shape[0]

    if take_average_over is None:
        take_average_over = 5
    
    if start_index is None:
        start_index = 0

    logging.info(f"averaging input over {take_average_over} bins")
    process_input(inputs, num_of_channels, take_average_over=take_average_over, start_index=start_index)

    return convert_to_tensor(inputs)

def fetch_output(data_folder, data_files):
    '''
    Fetch output (components). For use with older type of
    data saving, where simulation input parameters are kept in metadata
    file. The metadata file consists of a json-formatted "dictionary",
    with keys being the data files' names, and the values for these
    being the input parameter dictionaries. 

    Returns
    -------

    ### outputs : PyTorch tensor
        The components that created the simulation data.
    '''

    folder_path = os.path.join(
        os.getcwd(),
        data_folder
    )
    
    metadata = read_metadata(folder_path)
    outputs = get_components(metadata, data_files)
    return convert_to_tensor(outputs)

def model_training(
        train_data,
        test_data,
        model:Model,
        epochs,
        tolerance,
    ):

    '''
    


    Parameters
    ----------

    ### train_data/test_data : list (length 2) of tensors
        Train/test input and output.


    Returns
    -------

    ### losses : list of floats
        The losses for each epoch.

    ### r2_scores : NumPy array of of arrays of floats
        Each element is an array of 
        (epoch, train r-squared, test r-squared). These are calculated
        every 10 epochs.
    
    ### best_model_state_dict : PyTorch model's state_dict
        The "best" state dict of the model. Best in this case
        means the model that had the highest test r-squared score. Note
        that the r-squared score is only calculated every 10 epochs.
    '''

    from copy import deepcopy

    x_train, y_train = train_data
    x_test, y_test = test_data

    # excepts a data set greater than 20. A larger batch size
    # can speed up training by making each epoch take less time,
    # but equally a smaller batch size might reach a good score
    # earlier in epochs compared to a larger batch size.
    batch_size = y_train.shape[0]//20
    logging.info(f"\nBatch size: {batch_size}")

    # mainly for checking initial state
    logging.info("Initial state sanity check")
    logging.info("==========================")
    model.evaluate(x_train[:15,], y_train[:15,], batch_size)
    logging.info("==========================")

    # Do optimisation
    #-------------------------------------

    #TODO: Some variable learning rate could be added. Best option
    # is likely the pytorch LR schedulers, particularly
    # ReduceLROnPlateu. 

    losses = [0]*epochs
    logging.info(f"\nEpochs: {epochs}")
    logging.info(f"tolerance: {tolerance}")
    previous_loss = np.inf
    previous_test_r2 = -np.inf
    r2_scores = []
    start = time.time()
    for epoch in range(epochs):
        print("\033[K",end="")
        print(f"{epoch+1}/{epochs}", end="\r")

        current_loss = model.train(x_train, y_train, batch_size)
        losses[epoch] = current_loss

        # save r2 evaluations, keep track of best model parameters
        if 0 == epoch%10:

            train_r2 = model.r2_evaluate(x_train, y_train)
            test_r2 = model.r2_evaluate(x_test,y_test)

            r2_scores.append(
                np.array([float(epoch)]+[train_r2,test_r2])
            )
            if previous_test_r2 < test_r2:
                previous_test_r2 = test_r2
                best_model_state_dict = deepcopy(model.inner_model.state_dict())

        # tolerance
        if abs(current_loss-previous_loss)/previous_loss <= tolerance:
            losses = losses[:epoch]
            break

        previous_loss = current_loss

    # ===========================

    stop = time.time()
    logging.info(f"Fitting took {stop-start} seconds")
    logging.info(f"Total epochs run: {epoch+1}")
    logging.info(f"Final loss: {current_loss}")

    return losses, r2_scores, best_model_state_dict

def plot_training_results(losses, r2_scores):

    fig, ax = plt.subplots(2,1)

    ax[0].plot(np.arange(len(losses))+1,losses)
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("epoch loss")
    ax[0].set_yscale("log")


    # first column has epoch, other two have train and test r2
    r2_scores = np.array(r2_scores)
    test_r2 = r2_scores[:,2]
    train_r2 = r2_scores[:,1]
    epoch_r2 = r2_scores[:,0]

    # include only scores greater than zero, if there are positive scores
    r2_geq_zero = np.nonzero(test_r2 >= 0)
    if np.any(r2_geq_zero):
        train_r2 = r2_scores[:,1][r2_geq_zero]
        epoch_r2 = r2_scores[:,0][r2_geq_zero]
        test_r2 = test_r2[r2_geq_zero]

    logging.info("Max test R2:")
    logging.info(test_r2.max())

    ax[1].plot(epoch_r2, train_r2, label="Train R2")
    ax[1].plot(epoch_r2, test_r2, label="Test R2")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("r2 score")
    ax[1].legend()
    ax[1].grid()
    plt.show()



def main(
        data_folder:str,
        train_size:int,
        test_size:int,
        epochs:int,
        tol:float,
        learning_rate=None,
        save_model=None

):
    '''
    Performs the training process of a model.

    Parameters
    ----------

    ### data_folder : string
        The name of the folder where the data resides.
    
    ### train_size, test_size : int
        Number of the training samples and test samples to use,
        respectively. Obviously depends on the number of simulated
        spectra - how many data files there are.

    ### epochs : int
        The number of epochs to at most train the model for. The true
        number of epochs run depends on <tol>, but is limited to
        <epochs> epochs.

    ### tol : float
        The minimum relative change in the loss to tolerate before
        ending the training. Probably best left quite small, and prefer
        setting a good <epochs> count: there's (presumably) no harm
        in running the training for "too" long, especially as the
        best model (based on test set score) is chosen anyway.

    ### learning_rate : float, default None
        The learning rate used by the optimiser. If None, defaults to
        0.005.

    ### save_model : Boolean, default None
        Determines whether the trained model will be saved for later
        use. 
        
        If True or None (default), the model state dictionary 
        along with some other parameters will be saved to
        a folder "saved_models" using `torch.save()`, with a file name 
        determined by the start time of the training. The model can be
        loaded from there using `torch.load()`: see the module
        "evaluate.py" for how this happens exactly.

        If False, will not save the model.
    '''

    # setup logger
    # -----------------
    log_folder = os.path.join(
        os.getcwd(),
        "logged_runs_pt"
    )

    if not (os.path.exists(log_folder)):
        os.mkdir(log_folder)

    date_str = helpers.date_time_str()
    file_name = "fit" + date_str + ".log"
    file_path = os.path.join(log_folder, file_name)

    handlers = logging.StreamHandler(sys.stdout), logging.FileHandler(file_path, encoding="utf-8")

    logging.basicConfig(
        style="{",
        format="{message}",
        level=logging.INFO,
        handlers=handlers
    )
    # =========================

    # Get inputs and outputs, process
    # --------------------------------------

    print("Starting data fetch and processing...")
    start = time.time()

    logging.info(f"Using data from folder {data_folder}")
    train_files, test_files = get_data_files(
        data_folder,
        train_size,
        test_size,
    )

    data_path = os.path.join(
        os.getcwd(),
        data_folder
    )
    x_train, y_train = get_simdata(data_path, train_files)
    x_test, y_test = get_simdata(data_path, test_files)

    print("Fetched input and output...")

    # Averaging over the input data. Could make this more easily
    # modifiable as well.
    take_average_over = 5
    start_index = 0
    num_of_channels = len(x_train[0])
    logging.info(f"averaging input over {take_average_over} bins")
    process_input(x_train, num_of_channels, take_average_over=take_average_over, start_index=start_index)
    process_input(x_test, num_of_channels, take_average_over=take_average_over, start_index=start_index)

    x_train = convert_to_tensor(x_train)
    x_test = convert_to_tensor(x_test)

    print("Processed input...")

    y_train = convert_to_tensor(y_train)
    y_test = convert_to_tensor(y_test)

    # for training on just on component
    # comp = 4
    # y_train = y_train[:,comp].reshape((-1,1))
    # y_test = y_test[:,comp].reshape((-1,1))

    # normalise outputs based on train output (could be problematic
    # if values in y_test are larger than in y_train, as the idea
    # would be to normalise to one?)
    y_train_col_max = y_train.amax(dim=0)
    y_train /= y_train_col_max
    y_test /= y_train_col_max

    print("Processed output.\n")

    end = time.time()
    print("Data fetch and processing took", end-start, " seconds.")
    # ======================================

    # Define the model
    # --------------------------------------

    input_size = x_train[0].shape[0]
    output_size = y_train[0].shape[0]

    layer_sizes = []
    layer_sizes.append(input_size)
    # decrease hidden layer size each layer 
    hidden_layer_sizes = [150-i*15 for i in range(10)]
    # hidden_layer_sizes = [150]*10
    layer_sizes.extend(hidden_layer_sizes)
    layer_sizes.append(output_size)

    logging.info(f"\nlayer sizes: {layer_sizes}")

    mlp = MLP(layer_sizes)

    if learning_rate is None:
        learning_rate = 0.005


    model = Model(
        mlp,
        torch.optim.Adam(mlp.parameters(), lr=learning_rate),
        torch.nn.MSELoss(),
    )

    logging.info("\nUsed optimiser:")
    logging.info(model.optim)
    # ======================================

    # Do training
    # --------------------------------------
    print("Beginning training...")

    losses, r2_scores, best_model_state_dict = model_training(
        (x_train, y_train),
        (x_test, y_test),
        model,
        epochs,
        tol
    )
    # ======================================

    # save model for easy use later
    # --------------------------------------

    if save_model is None:
        save_model = True

    if save_model:

        whole_state_dict = {
            "model_layers": layer_sizes,
            "model_state_dict": best_model_state_dict,
            "normalisation": y_train_col_max
        }

        save_model_state_dict(whole_state_dict,date_str)
    # ======================================

    plot_training_results(losses, r2_scores)


if __name__ == "__main__":
    # main(
    #     data_folder="simdata_train02",
    #     train_size=1900,
    #     test_size=200,
    #     epochs=100,
    #     tol=1e-8,
    #     learning_rate=0.001,
    #     save_model=False
    # )

    main(
        data_folder="temp_file",
        train_size=90,
        test_size=10,
        epochs=100,
        tol=1e-8,
        learning_rate=0.001,
        save_model=False
    )