import torch
import numpy as np
import logging
#pylint: disable=logging-not-lazy, logging-fstring-interpolation
import time
import os
import sys

import matplotlib.pyplot as plt

from MLPRegressor_test import process_input
from read_data import read_metadata

# Useful examples:
# https://colab.research.google.com/github/bentrevett/pytorch-image-classification/blob/master/1_mlp.ipynb
# https://discuss.pytorch.org/t/how-to-create-mlp-model-with-arbitrary-number-of-hidden-layers/13124


class MLP(torch.nn.Module):
    '''
    

    layer_sizes
        first should be input size, last
        shoud be output size, in between are
        the sizes of the hidden layers.
    '''

    def __init__(self, layer_sizes):

        super().__init__()

        self.layers = torch.nn.ModuleList()

        for i in range(len(layer_sizes)-1):
            inf = layer_sizes[i]
            outf = layer_sizes[i+1]
            self.layers.append(
                torch.nn.Linear(in_features=inf, out_features=outf)
            )


    def forward(self, x):

        for layer in self.layers[:-1]:
            x = torch.nn.functional.relu(layer(x))

        x = self.layers[-1](x)

        return x

def train(model, inputs, outputs, batch_size, optim, loss_func):
    '''
    inputs : torch tensor?
    '''

    epoch_loss = 0

    model.train()

    # TODO: could make a proper dataloader thing for this
    num_batches = inputs.shape[0]//batch_size

    for i in range(num_batches):
        x = inputs[i*batch_size:(i+1)*batch_size,:]
        y = outputs[i*batch_size:(i+1)*batch_size,:]

        # print("x:")
        # print(x)
        # print("y:")
        # print(y)

        optim.zero_grad()

        pred = model(x)

        # print(pred)

        loss = loss_func(pred, y)
        loss.backward()
        epoch_loss += loss.item()

        optim.step()

    return epoch_loss/num_batches

def train_with_closure(model, inputs, outputs, batch_size, optim, loss_func):
    '''
    Used with optimisers that require passsing closure.
    '''

    # TODO: this doesn't really work. The PyTorch documentation is
    # rather vague on how this is supposed to be done.

    model.train()

    epoch_loss = 0

    # TODO: could make a proper dataloader thing for this
    num_batches = inputs.shape[0]//batch_size

    for i in range(num_batches):
        x = inputs[i*batch_size:(i+1)*batch_size,:]
        y = outputs[i*batch_size:(i+1)*batch_size,:]

        # print("x:")
        # print(x)
        # print("y:")
        # print(y)

        def closure():

            optim.zero_grad()

            pred = model(x)

            # print(pred)

            loss = loss_func(pred, y)
            loss.backward()
            epoch_loss += loss.item()


            return loss

        optim.step(closure)

    return epoch_loss/num_batches


def r2_score(pred,y):

    return (1-
            ((y-pred)**2).sum()/
            ((y-y.mean(dim=0))**2).sum()
        )


def r2_evaluate(model, x_train, y_train, x_test, y_test, do_logging=False):
    '''
    Evaluates the R2 score, optionally does logging to log the values.

    Returns
    -------

    ### train_r2, test_r2
    '''

    model.eval()
    with torch.no_grad():

        train_r2 = r2_score(model(x_train), y_train).item()
        test_r2 = r2_score(model(x_test), y_test).item()

    if do_logging:
        logging.info("\nR2:")
        logging.info("train evaluation:")
        logging.info(train_r2)
        logging.info("test evaluation:")
        logging.info(test_r2)

    return [train_r2, test_r2]

def evaluate(model, inputs, outputs, batch_size, loss_func):


    with torch.no_grad():

        epoch_loss = 0

        model.eval()

        # TODO: could make a proper dataloader thing for this
        num_batches = inputs.shape[0]//batch_size
        if num_batches == 0:
            num_batches = 1

        for i in range(num_batches):
            x = inputs[i*batch_size:(i+1)*batch_size,:]
            y = outputs[i*batch_size:(i+1)*batch_size,:]

            pred = model(x)

            # logging.info("prediction, true and difference:")
            # logging.info(
            #     torch.stack((pred.flatten(),y.flatten(),(pred.flatten()-y.flatten()).abs()), dim=1)
            # )

            logging.info("prediction:")
            logging.info(pred)
            logging.info("true:")
            logging.info(y)
            logging.info("difference:")
            logging.info((pred-y).abs())

            loss = loss_func(pred, y)
            logging.info("loss:")
            logging.info(loss.item())
            logging.info("R2:")
            logging.info(r2_score(pred,y).item())
            epoch_loss += loss.item()

        logging.info("mean loss:")
        logging.info(epoch_loss/num_batches)
    


def model_training():

    from read_data import get_train_or_test
    from read_data import get_components

    from copy import deepcopy

    
    # setup logger
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

    rng = np.random.default_rng()

    start = time.time()

    # get training and test input, process them
    # --------------------------

    folder = "simdata_more_random3"
    logging.info(f"simdata folder: {folder}")
    folder_path = os.path.join(os.getcwd(), folder)
    data_files = os.listdir(folder_path)
    data_files.remove("metadata.txt")

    train_size = 3800
    test_size = 200

    logging.info(f"train size: {train_size}")

    permute = False
    if permute:
        perm_data_files = rng.permutation(data_files)
        train_files = perm_data_files[:train_size]
        test_files = perm_data_files[train_size:train_size+test_size]
    else:
        train_files = data_files[:train_size]
        test_files = data_files[train_size:train_size+test_size]

    x_train = get_train_or_test(folder_path, train_files)
    x_test = get_train_or_test(folder_path, test_files)

    num_of_channels = x_train[0].shape[0]
    take_average_over = 5
    start_index = 0
    logging.info(f"averaging input over {take_average_over} bins")
    process_input(x_train, num_of_channels, take_average_over=take_average_over, start_index=start_index)
    process_input(x_test, num_of_channels, take_average_over=take_average_over, start_index=start_index)


    # need to use float32 due to problems with mismatch, but should not be a problem
    conv_to_tensor = lambda val: torch.tensor(np.array(val), dtype=torch.float32)

    x_train = conv_to_tensor(x_train)
    x_test = conv_to_tensor(x_test)

    # ================================================

    # what to fit on: lifetimes, intensities or both ("all")
    output_to_fit = "all"
    logging.info(f"Fitting over {output_to_fit}")

    # get output data
    metadata = read_metadata(folder_path)
    y_train = get_components(metadata, train_files, output_to_fit)
    y_test = get_components(metadata, test_files, output_to_fit)

    y_train = conv_to_tensor(y_train)
    y_test = conv_to_tensor(y_test)

    # normalise to [0,1]
    y_train_col_max = y_train.amax(dim=0)
    y_train /= y_train_col_max
    y_test = y_test/y_train_col_max
    logging.info("\n[0,1] normalisation of output by dividing with")
    logging.info(y_train_col_max)

    
    # match output_to_fit:
    #     case "intensities":

    #         y_train *= 10
    #         y_test *= 10
    #         y_train[:,-1] *= 10
    #         y_test[:,-1] *= 10
    #     case "lifetimes":
    #         y_train /= 100
    #         y_test /= 100
    #         y_train[:,-1] /= 10
    #         y_test[:,-1] /= 10



    stop = time.time()
    logging.info("time to fetch and process test and train: " + str(stop-start))

    # ensure correct functioning by reshape
    # (this is for fitting only to one or some of the outputs)
    # comp = 2
    # y_train = y_train[:,comp].reshape((-1,1))
    # y_test = y_test[:,comp].reshape((-1,1))

    input_size = x_train[0].shape[0]
    output_size = y_train[0].shape[0]

    layer_sizes = []
    layer_sizes.append(input_size)
    # decrease hidden layer size each layer 
    hidden_layer_sizes = [150-i*15 for i in range(10)]
    layer_sizes.extend(hidden_layer_sizes)
    layer_sizes.append(output_size)

    logging.info(f"layer sizes: {layer_sizes}")

    model = MLP(layer_sizes)
    loss_func = torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.0005)
    # optim = torch.optim.LBFGS(model.parameters(), max_iter=5, lr=0.01)

    logging.info("Optimiser:")
    logging.info(optim)

    batch_size = train_size//10
    logging.info(f"\nBatch size: {batch_size}")

    evaluate(model, x_train[:15,], y_train[:15,], batch_size, loss_func)

    # Do optimisation
    #-------------------------------------

    epochs = 6000
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
        # current_loss = train_with_closure(model, x_train, y_train, batch_size, optim, loss_func)
        current_loss = train(model, x_train, y_train, batch_size, optim, loss_func)
        losses[epoch] = current_loss

        # save r2 evaluations, keep track of best model parameters
        if 0 == epoch%10:
            train_r2, test_r2 = r2_evaluate(model, x_train, y_train, x_test, y_test)
            r2_scores.append(
                np.array([float(epoch)]+[train_r2,test_r2])
            )
            if previous_test_r2 < test_r2:
                previous_test_r2 = test_r2
                best_model_state_dict = deepcopy(model.state_dict())

        # tolerance
        if abs(current_loss-previous_loss)/previous_loss <= tolerance:
            losses = losses[:epoch]
            break

        previous_loss = current_loss

    stop = time.time()
    logging.info(f"Fitting took {stop-start} seconds")

    # =======================================

    whole_state_dict = {
        "model_layers": layer_sizes,
        "model_state_dict": best_model_state_dict,
        "normalisation": y_train_col_max
    }
    # save model
    save_model_state_dict(
        whole_state_dict,
        date_str
    )
        
    logging.info(f"Total epochs run: {epoch+1}")

    # y_train *= y_train_col_max
    # y_test *= y_train_col_max

    r2_only = False
    if r2_only:
        r2_evaluate(model, x_train, y_train, x_test, y_test, do_logging=True)
    else:
        logging.info("\n train evaluation:")
        evaluate(model, x_train[:15,], y_train[:15,], batch_size, loss_func)
        logging.info("\n test evaluation:")
        evaluate(model, x_test[:15,], y_test[:15,], batch_size, loss_func)


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


    #TODO: test optimiser stuff, make adaptive learning rate? test
    # LBFGS again

def model_testing():

    from read_data import get_train_or_test
    from read_data import get_components

    from monte_carlo_PALS_TA import sim_pals

    from scipy.stats import kstest

    rng = np.random.default_rng()

    # get simulated data, process
    # --------------------------

    folder = "simdata_more_random3"
    folder_path = os.path.join(os.getcwd(), folder)
    data_files = os.listdir(folder_path)
    data_files.remove("metadata.txt")

    file_number = rng.integers(low=3805, high=4001)
    data_file = [data_files[file_number]]

    sim_x = get_train_or_test(folder_path, data_file)

    num_of_channels = sim_x[0].shape[0]
    take_average_over = 5
    start_index = 0
    process_input(sim_x, num_of_channels, take_average_over=take_average_over, start_index=start_index)


    # need to use float32 due to problems with mismatch, but should not be a problem
    conv_to_tensor = lambda val: torch.tensor(np.array(val), dtype=torch.float32)

    sim_x = conv_to_tensor(sim_x)

    # what the data was fit on: lifetimes, intensities or both ("all")
    output_to_fit = "all"
    # get output and process
    metadata = read_metadata(folder_path)
    sim_y = get_components(metadata, data_file, output_to_fit)
    sim_y = conv_to_tensor(sim_y)

    # ================================================


    # NOTE: real data is has a time gate of 10_000, which is not
    # what the simulated data the models have been trained on has.
    # Would need to train of different data
    # get real data
    # --------------------------------
    # real_folder = os.path.join(
    #     os.getcwd(),
    #     "Experimental_data20230215"
    # )

    # real_data_files = [file for file in os.listdir(real_folder) if file.endswith(".pals")]

    # real_data_file = [real_data_files[rng.integers(low=0, high=len(real_data_files))]]

    # real_metadata = read_metadata(real_folder, "metadata.json")
    # real_x = get_train_or_test(real_folder, real_data_file)
    # process_input(real_x, num_of_channels, take_average_over=take_average_over)
    # real_y = get_components(real_metadata, real_data_file, output_to_fit)

    # real_x = conv_to_tensor(real_x)
    # real_y = conv_to_tensor(real_y)
    # ========================================
    

    model_to_test = r"C:\Users\OMISTAJA\Desktop\Läksyt\2022-04\DSProject\Data_science_project_2023\saved_models\model20230413221636.pt"

    with open(model_to_test, "rb") as f:
        state_dict = torch.load(f)


    model = MLP(state_dict["model_layers"])
    model.load_state_dict(state_dict["model_state_dict"])
    model.eval()

    pred = model(sim_x).detach()*state_dict["normalisation"]

    # y_train_col_max = conv_to_tensor(
    #     [2.9490e+02, 8.4497e-01, 4.7999e+02, 2.9710e-01, 1.7999e+03, 3.9749e-02]
    # )

    pred = pred.flatten()
    sim_y = sim_y.flatten()

    print(pred)
    # so obviously the intensities are not currently constrained to
    # be anything specifically, as my idea was originally that I'd like
    # to first see if it can, on its own, get the right values. This
    # does seem to be close to one and at least not greater, but it
    # isn't so great to have it less than one either.
    print(pred[1::2].sum())
    print(sim_y)


    # TODO: have a "naive" prediction of just the mean components,
    # see how that compares to everything

    pred_components = [pred[2*i:2*i+2].tolist() for i in range(3)]
    pred_input = {"num_events": 1_000_000,
                    "bkg": 0.0,
                    "components": pred_components,
                    "bin_size": 25,
                    "time_gate": 15_000,
                    "sigma_start": 68,
                    "sigma_stop": 68,
                    "offset": 2000}
    pred_bins, pred_hist = sim_pals(pred_input, rng)

    sim_y_components = [sim_y[2*i:2*i+2].tolist() for i in range(3)]
    sim_y_input = {"num_events": 1_000_000,
                    "bkg": 0.0,
                    "components": sim_y_components,
                    "bin_size": 25,
                    "time_gate": 15_000,
                    "sigma_start": 68,
                    "sigma_stop": 68,
                    "offset": 2000}
    sim_y_bins, sim_y_hist = sim_pals(sim_y_input, rng)

    # here, the components are kind of mean values of the simulated
    # data.
    naive_input = {"num_events": 1_000_000,
                    "bkg": 0.0,
                    "components": [(245, 0.77), (400, 0.2095), (1500, 0.0205)],
                    "bin_size": 25,
                    "time_gate": 15_000,
                    "sigma_start": 68,
                    "sigma_stop": 68,
                    "offset": 2000}
    naive_bins, naive_hist = sim_pals(naive_input, rng)

    # pred_real = model(real_x).detach()*state_dict["normalisation"].flatten()
    # real_y = real_y.flatten()
    # pred_real_components = [pred_real[2*i:2*i+2].tolist() for i in range(3)]
    # pred_real_input = {"num_events": 1_000_000,
    #                 "bkg": 0.0,
    #                 "components": pred_real_components,
    #                 "bin_size": 25,
    #                 "time_gate": 15_000,
    #                 "sigma_start": 68,
    #                 "sigma_stop": 68,
    #                 "offset": 2000}
    # pred_real_bins, pred_real_hist = sim_pals(pred_real_input, rng)

    # real_y_components = [real_y[2*i:2*i+2].tolist() for i in range(3)]
    # real_y_input = {"num_events": 1_000_000,
    #                 "bkg": 0.0,
    #                 "components": real_y_components,
    #                 "bin_size": 25,
    #                 "time_gate": 15_000,
    #                 "sigma_start": 68,
    #                 "sigma_stop": 68,
    #                 "offset": 2000}
    # real_y_bins, real_y_hist = sim_pals(real_y_input, rng)


    print("\nTwo-sample kstest, sim data")
    print(kstest(sim_y_hist, pred_hist))
    print(np.abs(pred_hist-sim_y_hist).max())

    print("\nTwo-sample kstest, naive prediction against sim data")
    print(kstest(sim_y_hist, naive_hist))
    print(np.abs(naive_hist-sim_y_hist).max())


    plt.plot(pred_bins, pred_hist, label="predicted")
    plt.plot(sim_y_bins, sim_y_hist, linestyle="dashed", label="true")
    plt.title("Prediction versus true spectrum, randomly chosen simulation data (validation set)")



    # plt.plot(naive_bins, naive_hist, label="naive")
    # plt.plot(pred_bins, pred_hist-sim_y_hist, label="residual")

    # plt.plot(np.sort(pred_hist)/pred_hist.max(), label="predicted CDF")
    # plt.plot(np.sort(sim_y_hist)/sim_y_hist.max(), linestyle="dashed", label="true CDF")

    plt.legend()
    plt.xlabel("time")
    plt.ylabel("counts")
    plt.yscale("log")
    plt.show()



    # https://wiki.helsinki.fi/pages/viewpage.action?pageId=353490171
    
    



if __name__ == "__main__":

    model_testing()
