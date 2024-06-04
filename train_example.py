'''

# Instructions:

Everything that's needed to do here (to begin with) is contained in the arguments
of the `main()` function at the bottom. Main things to possibly
change in the function:

- <data_folder>: The folder where the simulated data was saved.

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

import torch

import helpers

from read_data import (
    get_data_files,
    get_simdata
)

from processing import (
    process_input01,
    SubMinDivByMax
)

from pytorch_helpers import (
    r2_score, 
    save_model_state_dict, 
    convert_to_tensor
)


from models import (
    PALS_GNLL, Conv1, PALS_GNLL_Intensities, PALS_GNLL_Single
    )

from train import Model, plot_training_results, model_training

import load_model


def define_gnll_model(
    true_size,
    learning_rate=None,
    device=None,
    dtype=None,
    model_state_dict=None,
    network_state_dict=None,
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

            r2_loss = ((1-r2_score(_input, true))).sum()

            return r2_loss + self._loss_func(_input, true, var)
        
        def get_predictions(self, x):
            
            return (x.normal.mean, x.softmax.mean)
    
    if not (model_state_dict is None):
        model = PALSModel.load_state_dict(model_state_dict)
        model.logging_info()
        return model, {"normal":lifetime_idx, "softmax":softmax_idx}

    linear = torch.nn.LazyLinear
    conv = Conv1
    pool = torch.nn.MaxPool1d
    # pool = torch.nn.AvgPool1d

    layers = [
        (conv(1,output_size,5,5), True),
        # (pool(4,4), False),
        (conv(output_size, output_size, 4,4), False),
        (conv(output_size,output_size*3,3,), True),
        # (conv(1,27,2,2), True),
        # (pool(4,4), False),
        # (conv(27,9,3,), True),
        # (conv(9,3,3,), True),
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
    # sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optim,
    #     factor=0.5,
    #     patience=100,
    #     verbose=True,
    #     threshold=0.01,
    #     mode="max",
    #     threshold_mode="abs",
    #     min_lr=learning_rate*0.5
    # )

    # sched = torch.optim.lr_scheduler.CyclicLR(
    #     optim,
    #     base_lr=learning_rate*0.5,
    #     max_lr=learning_rate,
    #     step_size_up=200,
    #     cycle_momentum=False
    # )

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
    return model,{"normal":lifetime_idx, "softmax":softmax_idx}

def define_gnll_model_intensities(
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
    # lifetime_idx = list(range(0,true_size-1,2))
    # intensity_idx = list(range(1,true_size-1,2))
    intensity_idx = list(range(true_size-1))
    bkg_idx = true_size-1
    softmax_idx = intensity_idx + [bkg_idx]
    # variance does not have correspondence in the "true" output,
    # but should the same size as the true output
    # lifetime_var_idx = list(range(true_size, output_size-1, 2))
    # intensity_var_idx = list(range(true_size+1, output_size-1, 2))
    intensity_var_idx = list(range(true_size, output_size-1))
    bkg_var_idx = output_size-1
    softmax_var_idx = intensity_var_idx + [bkg_var_idx]

    class PALSModel(Model):

        def transform_true(self, true):
            
            return true[:,softmax_idx]
        
        def loss_func(self, pred, true):
            
            # assume pred is (mean, var)
            _input, var = pred

            r2_loss = (1-r2_score(_input, true)).sum()

            return r2_loss + self._loss_func(_input, true, var)
        
        def get_predictions(self, x):
            
            return x[0]
    
    if not (model_state_dict is None):
        model = PALSModel.load_state_dict(model_state_dict)
        model.logging_info()
        return model, {"softmax":softmax_idx}

    linear = torch.nn.LazyLinear
    conv = Conv1
    pool = torch.nn.MaxPool1d

    output_size = 6

    layers = [
        (conv(1,output_size,5,5), True),
        # (pool(4,4), False),
        (conv(output_size, output_size, 4,4), False),
        (conv(output_size,output_size*3,3,), True),
        # (conv(1,27,2,2), True),
        # (pool(4,4), False),
        # (conv(27,9,3,), True),
        # (conv(9,3,3,), True),
        # (conv(9,27,3), True),
        (torch.nn.Flatten(), False),
        (linear(output_size*9), True),
        # (linear(output_size*3), True),
        (linear(2), False),
    ]

    if not (network_state_dict is None):
        network = load_model.load_network(network_state_dict, PALS_GNLL_Intensities)
    else:
        network = PALS_GNLL_Intensities(
            layers,
            [softmax_idx, softmax_var_idx]
        )

    if learning_rate is None:
        learning_rate = 0.0001

    optim = torch.optim.Adam(network.parameters(), lr=learning_rate)
    # TODO: check out CosineAnnealingWarmRestarts
    # sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optim,
    #     factor=0.5,
    #     patience=100,
    #     verbose=True,
    #     threshold=0.01,
    #     mode="max",
    #     threshold_mode="abs",
    #     min_lr=learning_rate*0.5
    # )

    # sched = torch.optim.lr_scheduler.CyclicLR(
    #     optim,
    #     base_lr=learning_rate*0.5,
    #     max_lr=learning_rate,
    #     step_size_up=200,
    #     cycle_momentum=False
    # )

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
    return model, {"softmax":softmax_idx}


def define_gnll_model_single(
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

    output_size = 2

    class PALSModel(Model):

        def transform_true(self, true):
            
            return true
        
        def loss_func(self, pred, true):
            
            # assume pred is (mean, var)
            _input, var = pred

            r2_loss = (1-r2_score(_input, true)).sum()

            return r2_loss + self._loss_func(_input, true, var)
        
        def get_predictions(self, x):
            
            return x[0]
    
    if not (model_state_dict is None):
        model = PALSModel.load_state_dict(model_state_dict)
        model.logging_info()
        return model, {"softmax":[]}

    linear = torch.nn.LazyLinear
    conv = Conv1

    output_size = 6

    layers = [
        (conv(1,output_size,5,5), True),
        # (pool(4,4), False),
        (conv(output_size, output_size, 4,4), False),
        (conv(output_size,output_size*3,3,), True),
        # (conv(1,27,2,2), True),
        # (pool(4,4), False),
        # (conv(27,9,3,), True),
        # (conv(9,3,3,), True),
        # (conv(9,27,3), True),
        (torch.nn.Flatten(), False),
        (linear(output_size*9), True),
        # (linear(output_size*3), True),
        (linear(2), False),
    ]

    if not (network_state_dict is None):
        network = load_model.load_network(network_state_dict, PALS_GNLL_Single)
    else:
        network = PALS_GNLL_Single(
            layers,
            [[0], [1]]
        )

    if learning_rate is None:
        learning_rate = 0.0001

    optim = torch.optim.Adam(network.parameters(), lr=learning_rate)

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
    return model, {"softmax":[]}

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


def perform_training(
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
        None, does not load a checkpoint. If empty string, load latest
        checkpoint. Precedes <network_state_checkpoint>: if this is given,
        the network is initialised based on the contents of this checkpoint,
        and <network_state_checkpoint> is ignored, as are other variables
        regarding the model, such as learning rate.


    ### network_state_checkpoint : str, default None
        Load a checkpoint for the neural network based on the filename <network_state_checkpoint>. If
        None, does not load a checkpoint. If empty string, load latest
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
    x_test, y_test, _ = get_simdata(data_path, test_files, inputs)

    def _in(words, _str):

        for word in words:
            if word in _str:
                return True
        return False
    
    # words = ["lifetime", "bkg", "1", "3"]
    # y_train = np.array([comp for comp,name in zip(y_train.T, comp_names) if not _in(words, name)]).T
    # y_test = np.array([comp for comp,name in zip(y_test.T, comp_names) if not _in(words, name)]).T


    output_size = y_train[0].shape[0]


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

    print("x_train min:", x_train.min())
    print("x_test min:", x_test.min())

    y_train = convert_to_tensor(y_train, device=dev, dtype=dtype)
    y_test = convert_to_tensor(y_test, device=dev, dtype=dtype)

    # comp_idx = 2
    # y_train = y_train[:,comp_idx].reshape((-1,1))
    # y_test = y_test[:,comp_idx].reshape((-1,1))

    # for training on just one component
    # comp = 4
    # y_train = y_train[:,comp].reshape((-1,1))
    # y_test = y_test[:,comp].reshape((-1,1))

    print("Component variation:")
    print(y_train.amin(dim=0))
    print(y_train.amax(dim=0))

    print("Processed output.\n")

    end = time.time()
    print("Data fetch and processing took", end-start, " seconds.")
    # ======================================

    # Define the model
    # --------------------------------------

    if isinstance(network_state_checkpoint, str):
        val = network_state_checkpoint
        if len(network_state_checkpoint) == 0: #load latest
            val = None
        network_state = load_model.load_train_dict(model_file=val)
    else:
        network_state = None
    
    if isinstance(model_state_checkpoint, str):
        val = model_state_checkpoint
        if len(model_state_checkpoint) == 0: #load latest
            val = None
        model_state = load_model.load_train_dict(model_file=val)
    else:
        model_state = None

    model, idx = define_gnll_model(
        output_size,
        learning_rate,
        device=dev,
        dtype=dtype,
        model_state_dict=model_state,
        network_state_dict=network_state,
    )

    # ======================================

    softmax_idx = idx["softmax"]
    if len(softmax_idx) == 0:
        softmax_idx = None
    proc = SubMinDivByMax
    # normalise outputs based on train output (could be problematic
    # if values in y_test are larger than in y_train, as the idea
    # would be to normalise to one?)
    output_processing = proc(y_train, no_processing_idx=softmax_idx)
    y_train = output_processing.process(y_train)
    y_test = output_processing.process(y_test)


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
            "output_normalisation": (type(output_processing), output_processing.state_dict()),
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

    # print(model.logging_info())

    plot_training_results(losses, r2_scores)


if __name__ == "__main__":


    torch.manual_seed(1000)

    # might want to try using a better loss,
    # right now it seems that the loss on the intensities and background
    # is far less than on the lifetimes, yet the R2 is worse for the
    # former

    # peform_training(
    #     data_folder="simdata_train14_no_noise",
    #     train_size=19000,
    #     test_size=1000,
    #     epochs=3000,
    #     tol=float("nan"),
    #     learning_rate=0.001,
    #     save_model=True,
    #     monitor=True,
    #     # model_state_checkpoint="model20240413172018.pt"
    # )



    perform_training(
        data_folder="simdata_train11",
        train_size=3000,
        test_size=1000,
        epochs=1000,
        tol=float("nan"),
        learning_rate=0.001,
        save_model=False,
        monitor=True,
    )