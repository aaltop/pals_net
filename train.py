'''
Utilities for training a model
'''

import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import logging
#pylint: disable=logging-not-lazy, logging-fstring-interpolation
import time

import numpy as np
import matplotlib.pyplot as plt
import torch

from pytorch_helpers import (
    r2_score, 
    pretty_print
)

from active_plot import active_plotting

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
            scheduler = sched_class(optim)
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

        ### pred
            Should be a tuple only if it is to contain separate
            values for prediction, such as ([normal, normal_var], [softmax, softmax_var]).
            Each item in the tuple will then have a loss calculated on it
            separately. Otherwise, should, for example, be a list,
            such as [normal, normal_var].
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

        self.inner_model.train()

        # TODO: could make a proper dataloader thing for this
        num_batches = inputs.shape[0]//batch_size

        # do at least one iteration
        if num_batches == 0:
            num_batches == 1

        epoch_loss = 0
        for i in range(num_batches):
            x = inputs[i*batch_size:(i+1)*batch_size,:]
            y = self.transform_true(outputs[i*batch_size:(i+1)*batch_size,:])


            self.optim.zero_grad()

            pred = self.inner_model(x)

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
    
    def r2_evaluate(self, inputs, outputs, do_logging=False, separate=False):
        '''
        Evaluates the R-squared score, optionally does logging to log the values.

        With <separate> set to True, returns the scores as separate values,
        rather than as their mean.

        Returns
        -------

        ### r2
        '''

        # for jankily testing mse loss vs. r2 loss
        # r2_score = lambda x,y: -torch.nn.functional.mse_loss(x,y, reduction="none").mean(dim=0)

        self.inner_model.eval()
        with torch.no_grad():
            
            pred = self.get_predictions(self.inner_model(inputs))
            if isinstance(pred, tuple):
                matches = zip(pred, self.transform_true(outputs))
                r2 = torch.cat([r2_score(pred_i, true_i) for pred_i, true_i in matches])
            else:
                r2 = r2_score(pred, outputs)


        if do_logging:
            logging.info("\nR-squared:")
            logging.info(r2)

        if separate:
            return r2
        else:
            return r2.mean().item()
    
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
    test_r2_separates = []
    start = time.time()
    monitoring_initialized = False
    for epoch in range(epochs):
        print("\033[K",end="")
        print(f"{epoch+1}/{epochs}", end="\r")

        rand_idx = torch.randperm(len(x_train))
        current_loss = model.train(x_train[rand_idx], y_train[rand_idx], batch_size)
        losses[epoch] = current_loss

        # save r2 evaluations, keep track of best model parameters
        if 0 == epoch%10:

            train_r2 = model.r2_evaluate(x_train, y_train)
            test_r2_separate = model.r2_evaluate(x_test,y_test, separate=True)
            test_r2 = test_r2_separate.mean().item()

            r2_scores.append(
                np.array([float(epoch)]+[train_r2,test_r2])
            )
            test_r2_separates.append(test_r2_separate.numpy(force=True))
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
            if monitor and epoch > 0 and 0 == epoch%50:
                x,train_y,test_y = np.array(r2_scores).T
                separates_array = np.array(test_r2_separates)

                # plot only positive r2 scores
                r2_geq_zero = np.nonzero(test_y >= 0)
                r2_x = x[r2_geq_zero]
                separates_array = separates_array[r2_geq_zero]

                plot_loss = False

                axes_mosaic = [
                    ["test R2 separate"],
                    ["Both R2"]
                ]
                if plot_loss:
                    axes_mosaic = [["loss"]] + axes_mosaic
                # assume a number of r2 values in separates_array each
                # in one column, so transpose to iterate over each
                # individual one
                separates_to_plot = [(r2_x, val) for val in separates_array.T]

                loss_to_plot = losses[:epoch]
                negative_losses = np.any(loss_to_plot < 0)
                to_plot = [
                    separates_to_plot,
                    [
                        (r2_x,test_y[r2_geq_zero]),
                        (r2_x, train_y[r2_geq_zero])
                    ]
                ]
                if plot_loss:
                    to_plot = [[(range(epoch),loss_to_plot)]] + to_plot

                if not monitoring_initialized:
                    fig, axs = plt.subplot_mosaic(axes_mosaic)
                    axs = list(axs.values())

                    for i in range(len(to_plot)):
                        for j in range(len(to_plot[i])):
                            axs[i].plot(*to_plot[i][j])
                            axs[i].grid(visible=True)


                    if plot_loss and not negative_losses:
                        axs[0].set_yscale("log")
                    plt.show(block=False)
                    monitoring_initialized = True

                

                active_plotting(axs, to_plot)
                plt.pause(0.05)

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
    model.r2_evaluate(x_test, y_test, do_logging=True)

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