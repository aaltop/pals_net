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

import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import logging
#pylint: disable=logging-not-lazy, logging-fstring-interpolation
import time
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
    process_input_mlp,
    process_input01
)

from pytorch_helpers import (
    r2_score, 
    save_model_state_dict, 
    convert_to_tensor,
    pretty_print
)

from active_plot import active_plotting

from models import MLP, NeuralNet, Conv1, PALS_MSE, PALS_GNLL

import load_model

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

    ### loss_kwargs : dictionary
        values passed to the loss when initialising.

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

    def __init__(self, model, optim, loss_func, loss_kwargs, scheduler=None, device=None, dtype=None):

        self.inner_model = model
        self.optim = optim
        self._loss_func = loss_func
        self.loss_kwargs = loss_kwargs
        self.scheduler = scheduler
        self.device = "cpu" if device is None else device
        self.dtype = torch.float32 if dtype is None else dtype

        self.to_device(self.inner_model)
        self.to_device(self._loss_func)

    def state_dict(self, best_inner_state=None):
        '''
        Return the state of the instance as a dictionary, to be
        loaded using `.load_state_dict()`.

        If <best_inner_state> is None, use current state dict of the inner
        model, else use <best_inner_state>.
        '''

        inner_state = self.inner_model.state_dict() if best_inner_state is None else best_inner_state

        state_dict = {
            "model":(type(self.inner_model), self.inner_model.instantiation_kwargs, inner_state),
            "optim":(type(self.optim), self.optim.state_dict()),
            "loss_func":(type(self._loss_func), self.loss_kwargs),
            "device":self.device,
            "dtype":self.dtype
        }

        if not (self.scheduler is None):
            state_dict["scheduler"] = (type(self.scheduler), self.scheduler.state_dict())

        return state_dict
    
    @classmethod
    def load_state_dict(cls, state_dict):

        model_class, model_kwargs, model_state = state_dict["model"]
        model = model_class(**model_kwargs)
        model.load_state_dict(model_state)

        optim_class, optim_state = state_dict["optim"]
        optim = optim_class(model.parameters())
        optim.load_state_dict(optim_state)

        loss_class, loss_kwargs = state_dict["loss_func"]
        loss_func = loss_class(**loss_kwargs)

        if "scheduler" in state_dict:
            sched_class, sched_state = state_dict["scheduler"]
            scheduler = sched_class()
            scheduler.load_state_dict(sched_state)
        else:
            scheduler = None

        return cls(model, optim, loss_func, loss_kwargs, scheduler, state_dict["device"], state_dict["dtype"])

    def to_device(self, arg):
        '''
        Move <arg> to the used device and dtype.
        '''

        return arg.to(self.device, self.dtype)
    
    
    def transform_true(self, true):
        '''
        Transform the true "output" <true> to match how `self.get_predictions()`
        outputs the predictions.
        '''

        return true
    
    def get_predictions(self, x):
        '''
        Given <x> as output from the prediction of `self.inner_model()`,
        return only the predictions.
        Mostly important for situations when a model outputs more
        than just the predictions, such as when both means and variances
        are output, and variances are not necessarily of interest in
        all situations. As such, "predictions" would in this case refer
        to the means, which would be the predictions of the true values,
        while variances would be the "confidence" of those predictions.

        Use self.inner_model(x) to get all parts of the prediction
        of the inner_model.
        '''

        return x
    
    def loss_func(self, pred, true):
        '''
        Wrapper for the inner loss function, for handling unusual
        prediction output from the inner model.
        '''

        return self._loss_func(pred, true)
    
    def calculate_loss(self, pred, true):
        '''
        Calculate the loss based on <pred>, the output from the neural
        net, and <true>, the true outputs, and return the loss.
        '''


        if isinstance(pred, tuple):
            loss = None
            for pred_i, true_i in zip(pred, true):
                if loss is None:
                    loss = self.loss_func(pred_i, true_i)
                else:
                    loss += self.loss_func(pred_i, true_i)
        else:
            loss = self.loss_func(pred, true)

        return loss

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
            y = self.transform_true(outputs[i*batch_size:(i+1)*batch_size,:])


            self.optim.zero_grad()

            pred = self.inner_model(x)

            # print(pred)
            loss = self.calculate_loss(pred, y)
            loss.backward()
            epoch_loss += loss.item()

            self.optim.step()

        average_loss = epoch_loss/num_batches

        if not (self.scheduler is None):
            r2 = self.r2_evaluate(inputs, outputs)
            self.scheduler.step(r2)

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
                y = self.transform_true(outputs[i*batch_size:(i+1)*batch_size,:])

                all_pred = self.inner_model(x)

                loss = self.calculate_loss(all_pred, y).item()
                logging.info("loss:")
                logging.info(loss)
                epoch_loss += loss

                # in case predictions are more than one tensor
                if isinstance(all_pred, tuple):
                    pred_and_true = zip(self.get_predictions(all_pred), y)
                else:
                    pred_and_true = ((self.get_predictions(all_pred), y),)

                for pred_i, true_i in pred_and_true:

                    logging.info("prediction:")
                    logging.info(pred_i)

                    logging.info("true:")
                    logging.info(true_i)

                    logging.info("difference:")
                    logging.info((pred_i-true_i).abs())


                    logging.info("R2:")
                    logging.info(r2_score(pred_i,true_i).mean().item())


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
            
            pred = self.get_predictions(self.inner_model(inputs))
            if isinstance(pred, tuple):
                matches = zip(pred, self.transform_true(outputs))
                r2 = torch.cat([r2_score(pred_i, true_i) for pred_i, true_i in matches]).mean().item()
            else:
                r2 = r2_score(pred, outputs).mean().item()

        if do_logging:
            logging.info("\nR-squared:")
            logging.info(r2)

        return r2
    
    def logging_info(self):

    

        logging.info("\nUsed model:")
        logging.info(self.inner_model)


        logging.info(f"Using device {self.device}")
        logging.info(f"Using dtype {self.dtype}")
        

        logging.info("\nUsed optimiser:")
        logging.info(self.optim)
        logging.info("\nUsed scheduler:")
        logging.info(None if self.scheduler is None else pretty_print(self.scheduler))
        logging.info("Used loss:")
        logging.info(pretty_print(self._loss_func, self.loss_kwargs))

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

    # A larger batch size
    # can speed up training by making each epoch take less time,
    # but equally a smaller batch size might reach a good score
    # earlier in epochs compared to a larger batch size.
    batch_size = max(y_train.shape[0]//20,1)
    logging.info(f"\nBatch size: {batch_size}")

    # mainly for checking initial state
    logging.info("Initial state sanity check")
    logging.info("==========================")
    print(x_train.shape)
    model.evaluate(x_test[:15,], y_test[:15,], batch_size)
    logging.info("==========================")

    # Do optimisation
    #-------------------------------------

    # losses = [0]*epochs
    losses = np.zeros(epochs, float)
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
            if monitor and epoch > 0 and 0 == epoch%10:
                x,train_y,test_y = np.array(r2_scores).T

                r2_geq_zero = np.nonzero(test_y >= 0)
                r2_x = x[r2_geq_zero]

                loss_to_plot = losses[:epoch]
                negative_losses = np.any(loss_to_plot < 0)
                to_plot = [
                    [
                        (range(epoch),loss_to_plot)
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
                            axs[i].grid(visible=True)


                    if not negative_losses:
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

    logging.info("\n End evaluation")
    logging.info("==================")
    model.evaluate(x_test[:15,], y_test[:15,], batch_size)

    return losses, r2_scores, best_model_state_dict

def plot_training_results(losses, r2_scores):

    fig, ax = plt.subplots(2,1)

    ax[0].plot(np.arange(len(losses))+1,losses)
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("epoch loss")
    if not np.any(losses < 0):
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


    logging.info(f"Max test R2 at epoch {1+test_r2.argmax()*10}:")
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

def define_mse_model(
    output_size,
    learning_rate=None,
    device=None,
    dtype=None,
    state_dict=None,
):


    linear = torch.nn.LazyLinear
    conv = Conv1
    pool = torch.nn.MaxPool1d

    layers = [
        (conv(1,3,5,5), True),
        (conv(3,27,3,), True),
        (torch.nn.Flatten(), False),
        (linear(output_size*9), True),
        (linear(output_size), False),
    ]

    # "components" contains lifetime-intensity pairs, "bkg" should
    # scalar at the end of a row
    # these should be the values that correspond to those in the
    # "true" output
    lifetime_idx = list(range(0,output_size-1,2))
    intensity_idx = list(range(1,output_size-1,2))
    bkg_idx = output_size-1
    softmax_idx = intensity_idx + [bkg_idx]

    class PALSModel(Model):

        def transform_true(self, true):
            
            return (true[:,lifetime_idx], true[:,softmax_idx])
        
    if not (state_dict is None):
        model = PALSModel.load_state_dict(state_dict)
        model.logging_info()
        return model, idx
    
    idx = [lifetime_idx, softmax_idx]
    network = PALS_MSE(
        layers,
        idx
    )


    if learning_rate is None:
        learning_rate = 0.0001

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
        
    

    model = PALSModel(
        network,
        optim,
        loss,
        loss_kwargs,
        scheduler=sched,
        device=device,
        dtype=dtype
    )

    model.logging_info()
    # state_dict() doesn't seem to work for the losses, so need to do
    # this
    logging.info("Loss kwargs:")
    logging.info(loss_kwargs)

    # return also the index correspondence for finding the correct
    # corresponendences from the true output
    return model, idx

def define_gnll_model(
    true_size,
    learning_rate=None,
    device=None,
    dtype=None,
    model_state_dict=None,
    network_state_dict=None
):
    '''
    Parameters
    ----------

    ### true_size
        size (number) of the "true" values, values to be predicted.
    '''

    output_size = 2*true_size

    # "components" contains lifetime-intensity pairs, "bkg" should
    # scalar at the end of a row
    # these should be the values that correspond to those in the
    # "true" output
    lifetime_idx = list(range(0,true_size-1,2))
    intensity_idx = list(range(1,true_size-1,2))
    bkg_idx = true_size-1
    softmax_idx = intensity_idx + [bkg_idx]
    # variance does not have correspondence in the "true" output,
    # but should the same size as the true output
    lifetime_var_idx = list(range(true_size, output_size-1, 2))
    intensity_var_idx = list(range(true_size+1, output_size-1, 2))
    bkg_var_idx = output_size-1
    softmax_var_idx = intensity_var_idx + [bkg_var_idx]

    class PALSModel(Model):

        def transform_true(self, true):
            
            return (true[:,lifetime_idx], true[:,softmax_idx])
        
        def loss_func(self, pred, true):
            
            # assume pred is (mean, var)
            _input, var = pred
            return self._loss_func(_input, true, var)
        
        def get_predictions(self, x):
            
            return (x.normal.mean, x.softmax.mean)
    
    if not (model_state_dict is None):
        model = PALSModel.load_state_dict(model_state_dict)
        model.logging_info()
        return model, [lifetime_idx, softmax_idx]

    linear = torch.nn.LazyLinear
    conv = Conv1
    pool = torch.nn.MaxPool1d

    # TODO: test maxpool?
    layers = [
        (conv(1,output_size,5,5), True),
        (pool(4,4), False),
        (conv(output_size,output_size*3,3,), True),
        # (conv(3,9,3,), True),
        # (conv(9,27,3), True),
        (torch.nn.Flatten(), False),
        (linear(output_size*9), True),
        # (linear(output_size*3), True),
        (linear(output_size), False),
    ]

    if not (network_state_dict is None):
        network = load_model.load_network(network_state_dict, PALS_GNLL)
    else:
        network = PALS_GNLL(
            layers,
            [lifetime_idx, lifetime_var_idx, softmax_idx, softmax_var_idx]
        )

    if learning_rate is None:
        learning_rate = 0.0001

    optim = torch.optim.Adam(network.parameters(), lr=learning_rate)
    # TODO: check out CosineAnnealingWarmRestarts
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim,
        factor=0.5,
        patience=100,
        verbose=True,
        threshold=0.01,
        mode="max",
        threshold_mode="abs",
        min_lr=learning_rate*0.5
    )
    sched = None

    loss_kwargs = {
    }
    loss = torch.nn.GaussianNLLLoss(**loss_kwargs)
        
    
    model = PALSModel(
        network,
        optim,
        loss,
        loss_kwargs,
        scheduler=sched,
        device=device,
        dtype=dtype
    )

    model.logging_info()
    # state_dict() doesn't seem to work for the losses, so need to do
    # this
    logging.info("Loss kwargs:")
    logging.info(loss_kwargs)

    # return also the index correspondence for finding the correct
    # corresponendences from the true output
    return model, [lifetime_idx, softmax_idx]


def setup_logger(date_str):

    log_folder = os.path.join(
        os.getcwd(),
        "logged_runs_pt"
    )

    if not (os.path.exists(log_folder)):
        os.mkdir(log_folder)

    file_name = "fit" + date_str + ".log"
    log_file_path = os.path.join(log_folder, file_name)

    handlers = logging.StreamHandler(sys.stdout), logging.FileHandler(log_file_path, encoding="utf-8")

    logging.basicConfig(
        style="{",
        format="{message}",
        level=logging.INFO,
        handlers=handlers
    )

    return log_file_path


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
        input_preprocessing=True,
        monitor=False,
        model_state_checkpoint=None,
        network_state_checkpoint=None

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
        a value that depends on the model.

    ### save_model : Boolean, default None
        Determines whether the trained model will be saved for later
        use. 
        
        If True or None (default), the model state dictionary 
        along with some other parameters will be saved to
        a folder "saved_models" using `torch.save()`, with a file name 
        determined by the start time of the training. The model can be
        loaded from there using `torch.load()`: see the module
        "load_model.py" for how this happens exactly.

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
    
    ### input_preprocessing : boolean, default True.
        Whether to perform preprocessing on the input.

    ### monitor : boolean, default False
        Whether to continuously plot the training scores.

    ### model_state_checkpoint : str, default None
        Load a checkpoint of the entire model based on the filename <model_state_checkpoint>. If
        None, does not load a checkpoint. If "latest", load latest
        checkpoint. Precedes <network_state_checkpoint>: if this is given,
        the network is initialised based on the contents of this checkpoint,
        and <network_state_checkpoint> is ignored, as are other variables
        regarding the model, such as learning rate.


    ### network_state_checkpoint : str, default None
        Load a checkpoint for the neural network based on the filename <network_state_checkpoint>. If
        None, does not load a checkpoint. If "latest", load latest
        checkpoint. See <model_state_checkpoint> for further info.
        
    '''

    date_str = helpers.date_time_str()
    log_file_path = setup_logger(date_str)

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

    inputs = (
        "components",
        "bkg"
    )
    x_train, y_train, comp_names = get_simdata(data_path, train_files, inputs)
    output_size = y_train[0].shape[0]
    x_test, y_test, _ = get_simdata(data_path, test_files, inputs)

    print("Fetched input and output...")

    # Averaging over the input data. Could make this more easily
    # modifiable as well.
    if input_preprocessing:
        pass
        # take_average_over = 5
        # start_index = 0
        # num_of_channels = len(x_train[0])
        # logging.info(f"Averaging input over {take_average_over} bins")
        # process_input_mlp(x_train, num_of_channels, take_average_over=take_average_over, start_index=start_index)
        # process_input_mlp(x_test, num_of_channels, take_average_over=take_average_over, start_index=start_index)

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

    print(x_train[0])
    print(x_test[0])
    print(y_train[0])
    print(y_test[0])

    if input_preprocessing:
        process_input = process_input01
        x_train = process_input(x_train)
        x_test = process_input(x_test)

    print("Processed input...")

    y_train = convert_to_tensor(y_train, device=dev, dtype=dtype)
    y_test = convert_to_tensor(y_test, device=dev, dtype=dtype)

    # for training on just one component
    # comp = 4
    # y_train = y_train[:,comp].reshape((-1,1))
    # y_test = y_test[:,comp].reshape((-1,1))

    print("Processed output.\n")

    end = time.time()
    print("Data fetch and processing took", end-start, " seconds.")
    # ======================================

    # Define the model
    # --------------------------------------

    model_class = PALS_GNLL

    if isinstance(network_state_checkpoint, str):
        val = network_state_checkpoint
        if len(network_state_checkpoint) == 0:
            val = None
        network_state = load_model.load_train_dict(model_file=val)
    else:
        network_state = None
    
    if isinstance(model_state_checkpoint, str):
        val = model_state_checkpoint
        if len(model_state_checkpoint) == 0:
            val = None
        model_state = load_model.load_train_dict(model_file=val)
    else:
        model_state = None

    if model_class is PALS_MSE:
        model, idx = define_mse_model(
            output_size, 
            learning_rate, 
            device=dev, 
            dtype=dtype,
            model_state_dict=model_state,
            network_state_dict=network_state,
        )
    elif model_class is PALS_GNLL:
        model, idx = define_gnll_model(
            output_size,
            learning_rate,
            device=dev,
            dtype=dtype,
            model_state_dict=model_state,
            network_state_dict=network_state,
        )

    # stat = model.state_dict()
    # model = model.load_state_dict(model.state_dict())
    # ======================================

    softmax_idx = idx[1]
    # normalise outputs based on train output (could be problematic
    # if values in y_test are larger than in y_train, as the idea
    # would be to normalise to one?)
    y_train_col_max = y_train.amax(dim=0)
    # don't normalise intensities or background
    y_train_col_max[softmax_idx] = 1.0
    y_train /= y_train_col_max
    y_test /= y_train_col_max

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

        whole_state_dict = model.state_dict(best_inner_state=best_model_state_dict)

        state_extend = {
            "normalisation": y_train_col_max,
            "log_file_path":log_file_path,
            "idx":idx,
            "sim_inputs":inputs,
        }

        whole_state_dict.update(state_extend)

        # these are the values used when preprocessing the input,
        # useful to pass them on so they can also be done when
        # evaluating the model
        if input_preprocessing:
            # whole_state_dict["process_input_parameters"] = {
            #     "num_of_channels":num_of_channels, 
            #     "take_average_over":take_average_over, 
            #     "start_index":start_index
            # }
            whole_state_dict["process_input_parameters"] = {
                "func_name": process_input.__name__
            }
        else:
            whole_state_dict["process_input_parameters"] = None

        save_model_state_dict(whole_state_dict,date_str)
    # ======================================

    plot_training_results(losses, r2_scores)


if __name__ == "__main__":


    torch.manual_seed(1000)

    # TODO: might want to try using a better loss,
    # right now it seems that the loss on the intensities and background
    # is far less than on the lifetimes, yet the R2 is worse for the
    # former
    main(
        data_folder="simdata_train01",
        train_size=39500,
        test_size=500,
        epochs=200,
        tol=float("nan"),
        learning_rate=0.0005,
        save_model=False,
        monitor=True,
        model_state_checkpoint="model20240308172317.pt"
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
    #     train_size=900,
    #     test_size=100,
    #     epochs=300,
    #     tol=float("nan"),
    #     learning_rate=0.0001,
    #     save_model=False,
    #     monitor=True
    # )

    # main(
    #     data_folder="temp_file_int",
    #     train_size=29500,
    #     test_size=500,
    #     epochs=1000,
    #     tol=1e-18,
    #     learning_rate=0.0001,
    #     save_model=False,
    #     monitor=True
    # )