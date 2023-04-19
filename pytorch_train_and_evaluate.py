import torch
import numpy as np
import logging
#pylint: disable=logging-not-lazy, logging-fstring-interpolation
import time
import os
import sys

import matplotlib.pyplot as plt

from read_data import (
    read_metadata, 
    get_train_or_test,
    get_components,
    get_data_files
)

from pytorch_helpers import (
    r2_score, 
    save_model_state_dict, 
    convert_to_tensor
)

from pytorchMLP import MLP





class Model:
    '''

    For the training and evaluation of a pytorch model.
    
    Parameters
    ----------

    ### model : PyTorch model

    ### optim : PyTorch optimiser

    ### loss_func : PyTorch loss function
    
    '''

    def __init__(self, model:MLP, optim, loss_func):

        self.inner_model = model
        self.optim = optim
        self.loss_func = loss_func


    
    def train(self, inputs, outputs, batch_size):
        '''
        Does one iteration of model training.
        '''

        epoch_loss = 0

        self.model.train()

        # TODO: could make a proper dataloader thing for this
        num_batches = inputs.shape[0]//batch_size

        # do at least one iteration
        if num_batches == 0:
            num_batches == 1

        for i in range(num_batches):
            x = inputs[i*batch_size:(i+1)*batch_size,:]
            y = outputs[i*batch_size:(i+1)*batch_size,:]


            self.optim.zero_grad()

            pred = self.model(x)

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

            self.model.eval()

            # TODO: could make a proper dataloader thing for this
            num_batches = inputs.shape[0]//batch_size
            if num_batches == 0:
                num_batches = 1

            for i in range(num_batches):
                x = inputs[i*batch_size:(i+1)*batch_size,:]
                y = outputs[i*batch_size:(i+1)*batch_size,:]

                pred = self.model(x)

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
                logging.info(r2_score(pred,y).item())
                epoch_loss += loss.item()

            logging.info("mean loss:")
            logging.info(epoch_loss/num_batches)

        return epoch_loss
    
    def r2_evaluate(self, inputs, outputs, do_logging=False):
        '''
        Evaluates the R-squared score, optionally does logging to log the values.

        Returns
        -------

        ### train_r2, test_r2
        '''

        self.model.eval()
        with torch.no_grad():

            r2 = r2_score(self.model(inputs), outputs).item()

        if do_logging:
            logging.info("\nR-squared:")
            logging.info(r2)

        return r2
    

def process_input(data, num_of_channels=None, take_average_over=None, start_index=None):
    '''
    Cut off counts in <data> such that only data from the
    <start_index> onwards up to <num_of_channels> beyond the <start_index> 
    is considered, and takes non-rolling means of <take_average_over> values 
    of the result.

    Parameters
    ----------
    ### data : list of numpy arrays
        The counts for <data.size> simulated spectra.

    ### num_of_channels : int
        How many channels to take beyond the max channel,
        as in [max_chan:max_chan+num_of_channels]. Needs to be
        divisible by <take_average_over>. Default is 100.

    ### take_average_over : int
        how many channels to average over. <num_of_channels> needs
        to be divisible by this. Default is 5.

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
    Fetch and process the train and test input data.

    Parameters
    ----------

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

def fetch_output(folder_path, data_files):
    
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
    '''

    from copy import deepcopy

    # setup logger (TODO: maybe have a more sensible place for this?)
    # -----------------
    log_folder = os.path.join(
        os.getcwd(),
        "logged_runs_pt"
    )

    if not (os.path.exists(log_folder)):
        os.mkdir(log_folder)

    date_str = time.strftime("%Y%m%d%H%M%S")
    file_name = "fit" + date_str + ".log"
    file_path = os.path.join(log_folder, file_name)

    handlers = logging.StreamHandler(sys.stdout), logging.FileHandler(file_path, encoding="utf-8")

    logging.basicConfig(
        style="{",
        format="{message}",
        level=logging.INFO,
        handlers=handlers
    )

    logging.info("starting now")
    # =========================

    x_train, y_train = train_data
    x_test, y_test = test_data

    batch_size =y_train.shape[0]//10
    logging.info(f"\nBatch size: {batch_size}")

    # mainly for checking initial state
    model.evaluate(x_train[:15,], y_train[:15,], batch_size)

    # Do optimisation
    #-------------------------------------

    losses = [0]*epochs
    logging.info(f"\nEpochs: {epochs}")
    tolerance = 1e-8
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

    return losses, r2_scores, best_model_state_dict