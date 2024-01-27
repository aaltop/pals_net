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


import logging
#pylint: disable=logging-not-lazy, logging-fstring-interpolation
import time
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch

import helpers

from read_data import (
    get_data_files,
    get_simdata
)

from processing import (
    process_input
)

from pytorch_helpers import (
    r2_score, 
    save_model_state_dict, 
    convert_to_tensor,
    pretty_print
)

from active_plot import active_plotting

from models import MLP, NeuralNet, Conv1

_rng = np.random.default_rng()

# It's kind of a stupid name, but don't know what
# else to call it
class Model:
    '''

    For the training and evaluation of a pytorch model.
    
    Parameters
    ----------

    ### model : PyTorch model

    ### optim : PyTorch optimiser

    ### loss_func : PyTorch loss function

    ### scheduler : PyTorch learning rate scheduler, default None
        If None, no scheduler is used, meaning the learning rate
        specified for the optimiser is not adjusted.

        Used schedulers should be readily usable by just calling
        their `step()` method.

    ### device : string, default None
        The torch.device to use for computations, e.g. "cpu" for CPU
        or "cuda" for GPU. If None, defaults to "cpu".

    ### dtype : Pytorch dtype, default None
        Datatype to use
    
    '''

    def __init__(self, model, optim, loss_func, scheduler=None, device=None, dtype=None):

        self.inner_model = model
        self.optim = optim
        self.loss_func = loss_func
        self.scheduler = scheduler
        self.device = "cpu" if device is None else device
        self.dtype = torch.float32 if dtype is None else dtype

        self.to_device(self.inner_model)
        self.to_device(self.loss_func)


    def to_device(self, arg):
        '''
        Moves <arg> to the used device and dtype.
        '''

        return arg.to(self.device, self.dtype)
    
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

        average_loss = epoch_loss/num_batches

        if not (self.scheduler is None):
            self.scheduler.step(average_loss)

        return average_loss
    
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

def model_training(
        train_data,
        test_data,
        model:Model,
        epochs,
        tolerance,
        monitor=False
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

    ### monitor : boolean
        Whether to plot the score and update the plot in "real" time.
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
    print(x_train.shape)
    model.evaluate(x_train[:15,], y_train[:15,], batch_size)
    logging.info("==========================")

    # Do optimisation
    #-------------------------------------

    losses = [0]*epochs
    logging.info(f"\nEpochs: {epochs}")
    logging.info(f"tolerance: {tolerance}")
    previous_loss = np.inf
    previous_test_r2 = -np.inf
    best_model_state_dict = deepcopy(model.inner_model.state_dict())
    r2_scores = []
    start = time.time()
    monitoring_initialized = False
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

            # TODO: While this plotting works, it's not entirely ideal.
            # Might want to use the animation module of matplotlib for
            # something that probably works better. The issue there
            # is that there is no class (as far as I can tell) that
            # allows updating an animation at certain loop intervals,
            # only at certain times. The timed version might also
            # be alright, would have to test though.

            # continuous plot of training process
            if monitor and epoch > 0 and 0 == epoch%100:
                x,train_y,test_y = np.array(r2_scores).T

                r2_geq_zero = np.nonzero(test_y >= 0)
                r2_x = x[r2_geq_zero]

                to_plot = [
                    [
                        (range(epoch),losses[:epoch])
                    ],
                    [
                        (r2_x,test_y[r2_geq_zero]),
                        (r2_x, train_y[r2_geq_zero])
                    ]
                ]

                if not monitoring_initialized:
                    fig, axs = plt.subplots(2,1)

                    for i in range(len(to_plot)):
                        for j in range(len(to_plot[i])):
                            axs[i].plot(*to_plot[i][j])

                    axs[0].set_yscale("log")
                    plt.show(block=False)
                    monitoring_initialized = True

                

                active_plotting(axs, to_plot)
                plt.pause(0.01)

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


def define_mlp(input_size:int, output_size:int, hidden_layer_sizes=None) -> MLP:
    '''
    Create the multilayer perceptron.

    Parameters
    ----------

    ### input_size : int
    
    ### output_size : int

    ### hidden_layer_sizes : list of ints, default None
        The sizes of layers in between the input and output layer.
        If None, uses a "good" value.

    Returns
    -------

    ### model : MLP
        The model.
    '''

    layer_sizes = []
    layer_sizes.append(input_size)
    # decrease hidden layer size each layer 
    hidden_layer_sizes = [150-i*15 for i in range(10)]
    # hidden_layer_sizes = [150]*10
    layer_sizes.extend(hidden_layer_sizes)
    layer_sizes.append(output_size)

    return MLP(layer_sizes)



def main(
        data_folder:str,
        train_size:int,
        test_size:int,
        epochs:int,
        tol:float,
        learning_rate=None,
        save_model=None,
        device=None,
        dtype=None,
        input_preprocessing=False,
        monitor=False

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

    ### device : string, default None
        The name of the device to use, e.g. "cpu" for CPU or "cuda" for
        GPU.

        If None, test to see whether cuda is available, and uses
        CPU or GPU based on the test.

    ### dtype : Pytorch dtype, default None
        The datatype to use. If None, defaults to torch.float32. 
        
        NOTE: There seems to be some issue at least with torch.float16, 
        not sure what. Best to use torch.float32.
    
    ### input_preprocessing : boolean, default False.
        Whether to perform preprocessing on the input. This was mainly
        used with the MLP, and preprocessing might be better switched
        for just a convolution or averaging layer anyway.

    ### monitor : boolean, default False
        Whether to continuously plot the training scores.
        
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
    log_file_path = os.path.join(log_folder, file_name)

    handlers = logging.StreamHandler(sys.stdout), logging.FileHandler(log_file_path, encoding="utf-8")

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

    data_path = os.path.join(
        os.getcwd(),
        data_folder
    )

    logging.info(f"Using data from folder {data_folder}")
    logging.info(f"Train size: {train_size}")
    logging.info(f"Test size: {test_size}")
    train_files, test_files = get_data_files(
        data_path,
        train_size,
        test_size,
    )

    x_train, y_train = get_simdata(data_path, train_files)
    x_test, y_test = get_simdata(data_path, test_files)

    print("Fetched input and output...")

    # Averaging over the input data. Could make this more easily
    # modifiable as well.
    if input_preprocessing:
        take_average_over = 5
        start_index = 0
        num_of_channels = len(x_train[0])
        logging.info(f"Averaging input over {take_average_over} bins")
        process_input(x_train, num_of_channels, take_average_over=take_average_over, start_index=start_index)
        process_input(x_test, num_of_channels, take_average_over=take_average_over, start_index=start_index)

    # try changing to GPU
    if device is None:
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        else:
            dev = torch.device("cpu")
    else:
        dev = device

    if dtype is None:
        dtype = torch.float32

    x_train = convert_to_tensor(x_train, device=dev, dtype=dtype)
    x_test = convert_to_tensor(x_test, device=dev, dtype=dtype)

    print("Processed input...")

    y_train = convert_to_tensor(y_train, device=dev, dtype=dtype)
    y_test = convert_to_tensor(y_test, device=dev, dtype=dtype)

    # for training on just one component
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

    # network = define_mlp(input_size, output_size)

    linear = torch.nn.LazyLinear
    conv = Conv1
    pool = torch.nn.MaxPool1d

    layers = [
        (conv(1,3,5,5), True),
        (conv(3,27,3,), True),
        (torch.nn.Flatten(), False),
        (linear(output_size*9), True),
        (linear(output_size), False)
    ]
    
    network = NeuralNet(layers)

    logging.info("\nUsed model:")
    logging.info(network)

    if learning_rate is None:
        learning_rate = 0.005


    logging.info(f"Using device {dev}")
    logging.info(f"Using dtype {dtype}")

    optim = torch.optim.Adam(network.parameters(), lr=learning_rate)
    # TODO: check out CosineAnnealingWarmRestarts
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim,
        factor=0.9,
        patience=10,
        verbose=True,
        min_lr=0.0005
    )
    sched = None

    loss_kwargs = {

    }
    loss = torch.nn.MSELoss(**loss_kwargs)

    model = Model(
        network,
        optim,
        torch.nn.MSELoss(),
        scheduler=sched,
        device=dev,
        dtype=dtype
    )

    logging.info("\nUsed optimiser:")
    logging.info(model.optim)
    logging.info("\nUsed scheduler:")
    logging.info(None if sched is None else pretty_print(sched))
    logging.info("Used loss:")
    logging.info(pretty_print(loss, loss_kwargs))
    # ======================================

    # Do training
    # --------------------------------------
    print("Beginning training...")

    losses, r2_scores, best_model_state_dict = model_training(
        (x_train, y_train),
        (x_test, y_test),
        model,
        epochs,
        tol,
        monitor=monitor
    )
    # ======================================

    # save model for easy use later (The model loading itself
    # might be possible to do more easily using "TorchScript", 
    # but there is all these other bits that need saving too)
    # --------------------------------------

    if save_model or (save_model is None):



        whole_state_dict = {
            "model_kwargs": network.instantiation_kwargs,
            "model_state_dict": best_model_state_dict,
            "normalisation": y_train_col_max,
            "device": dev,
            "dtype": dtype,
            "log_file_path":log_file_path
        }

        # these are the values used when preprocessing the input,
        # useful to pass them on so they can also be done when
        # evaluating the model
        if input_preprocessing:
            whole_state_dict["process_input_parameters"] = {
                "num_of_channels":num_of_channels, 
                "take_average_over":take_average_over, 
                "start_index":start_index
            }
        else:
            whole_state_dict["process_input_parameters"] = None

        save_model_state_dict(whole_state_dict,date_str)
    # ======================================

    plot_training_results(losses, r2_scores)


if __name__ == "__main__":


    torch.manual_seed(1000)

    main(
        data_folder="simdata_train01",
        train_size=19500,
        test_size=500,
        epochs=10000,
        tol=1e-10,
        learning_rate=0.00025,
        save_model=True,
        monitor=True
    )


    # main(
    #     data_folder="simdata_train02",
    #     train_size=1900,
    #     test_size=200,
    #     epochs=100,
    #     tol=1e-8,
    #     learning_rate=0.001,
    #     save_model=False
    # )



    # main(
    #     data_folder="temp_file",
    #     train_size=90,
    #     test_size=10,
    #     epochs=300,
    #     tol=1e-8,
    #     learning_rate=0.001,
    #     save_model=False,
    #     monitor=False
    # )