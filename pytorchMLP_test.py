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
            # epoch_loss += loss.item()


            return loss

        optim.step(closure)

    return epoch_loss/num_batches


def r2_score(pred,y):

    return (1-
            ((y-pred)**2).sum()/
            ((y-y.mean(dim=0))**2).sum()
        )

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

            # print("x:")
            # print(x)
            # print("y:")
            # print(y)

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
        




def main():

    
    # setup logger
    # -----------------
    log_folder = os.path.join(
        os.getcwd(),
        "logged_runs_pt"
    )

    if not (os.path.exists(log_folder)):
        os.mkdir(log_folder)

    file_name = "fit" + time.strftime("%Y%m%d%H%M%S") + ".log"
    file_path = os.path.join(log_folder, file_name)

    handlers = logging.StreamHandler(sys.stdout), logging.FileHandler(file_path)

    logging.basicConfig(
        style="{",
        format="{message}",
        level=logging.INFO,
        handlers=handlers
    )

    logging.info("starting now")
    # =========================

    from read_data import get_train_or_test
    from read_data import get_components

    rng = np.random.default_rng()

    start = time.time()

    # get training and test input, process them
    # --------------------------

    folder = "simdata_more_random3"
    logging.info(f"simdata folder: {folder}")
    folder_path = os.path.join(os.getcwd(), folder)
    data_files = os.listdir(folder_path)
    data_files.remove("metadata.txt")

    train_size = 1900
    test_size = 10

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

    num_of_channels = 200
    take_average_over = 5
    process_input(x_train, num_of_channels, take_average_over=take_average_over)
    process_input(x_test, num_of_channels, take_average_over=take_average_over)

    # need to use float32 due to problems with mismatch, but should not be a problem
    conv_to_tensor = lambda val: torch.tensor(np.array(val), dtype=torch.float32)

    x_train = conv_to_tensor(x_train)
    x_test = conv_to_tensor(x_test)

    logging.info(f"averaging input over {take_average_over} bins")

    # what to fit on: lifetimes, intensities or both ("all")
    output_to_fit = "lifetimes"
    logging.info(f"Fitting over {output_to_fit}")

    metadata = read_metadata(folder_path)
    y_train = get_components(metadata, train_files, output_to_fit)
    y_test = get_components(metadata, test_files, output_to_fit)

    # normalise to [0,1]
    # y_train = conv_to_tensor(y_train)
    # y_train_col_max = y_train.amax(dim=0)
    # y_train /= y_train_col_max
    # y_test = conv_to_tensor(y_test)/y_train_col_max

    
    match output_to_fit:
        case "intensities":

            y_train = conv_to_tensor(y_train)*10
            y_test = conv_to_tensor(y_test)*10
            y_train[:,-1] *= 10
            y_test[:,-1] *= 10
        case "lifetimes":
            y_train = conv_to_tensor(y_train)/100
            y_test = conv_to_tensor(y_test)/100
            y_train[:,-1] /= 10
            y_test[:,-1] /= 10



    stop = time.time()
    logging.info("time to fetch and process test and train: " + str(stop-start))

    #ensure correct functioning by reshape
    y_train = y_train[:,-2].reshape((-1,1))
    y_test = y_test[:,-2].reshape((-1,1))

    input_size = x_train[0].shape[0]
    output_size = y_train[0].shape[0]

    layer_sizes = []
    layer_sizes.append(input_size)
    # decrease hidden layer size each layer
    hidden_layer_sizes = [150-i*20 for i in range(7)]
    layer_sizes.extend(hidden_layer_sizes)
    layer_sizes.append(output_size)

    model = MLP(layer_sizes)
    loss_func = torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    # optim = torch.optim.LBFGS(model.parameters(), max_iter=5, lr=0.01)

    batch_size = 100
    logging.info(f"\nBatch size: {batch_size}")

    evaluate(model, x_train[:15,], y_train[:15,], batch_size, loss_func)

    epochs = 2000
    losses = [0]*epochs
    logging.info(f"\nEpochs: {epochs}")
    start = time.time()
    tolerance = 1e-5
    previous_loss = np.inf
    for epoch in range(epochs):
        print("\033[K",end="")
        print(f"{epoch+1}/{epochs}", end="\r")
        # train_with_closure(model, x_train, y_train, 5, optim, loss_func)
        current_loss = train(model, x_train, y_train, batch_size, optim, loss_func)
        losses[epoch] = current_loss

        # tolerance
        if abs(current_loss-previous_loss)/previous_loss <= tolerance:
            losses = losses[:epoch]
            logging.info(f"Total epochs:{epoch}")
            break

        previous_loss = current_loss
        

        
    stop = time.time()
    logging.info(f"Fitting took {stop-start} seconds")

    # y_train *= y_train_col_max
    # y_test *= y_train_col_max

    logging.info("\n train evaluation:")

    evaluate(model, x_train[:15,], y_train[:15,], batch_size, loss_func)

    logging.info("\n test evaluation:")

    evaluate(model, x_test[:15,], y_test[:15,], batch_size, loss_func)

    plt.plot(np.arange(len(losses))+1,losses)
    plt.xlabel("epoch")
    plt.ylabel("epoch loss")
    plt.yscale("log")
    plt.show()





if __name__ == "__main__":

    main()