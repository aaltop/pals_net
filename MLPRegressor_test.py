from sklearn.neural_network import MLPRegressor
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# from read_data import read_data
from read_data import read_metadata
import time

import logging

# SEE
# Pietrow, Marek & Miaskowski, A.. (2023). 
# Artificial neural network as an effective tool to calculate parameters
# of positron annihilation lifetime spectra. 

# I guess I SHOULD just do as the thing says, but no. Don't know
# if logging is the best here anyway, but it's a quick thing, and
# the point is currently to always log anyway
#pylint: disable=logging-not-lazy, logging-fstring-interpolation

def process_input(data, num_of_channels=None, take_average_over=None):
    '''
    Cut off counts in <data> such that only data from the
    max value onwards up to <num_of_channels> beyond the max 
    is considered, and takes non-rolling means of five values 
    of the result.

    Parameters
    ----------
    data : list of numpy arrays
        The counts for <data.size> simulated spectra.

    num_of_channels : int
        How many channels to take beyond the max channel,
        as in [max_chan:max_chan+num_of_channels]. Needs to be
        divisible by <take_average_over>.

    take_average_over : int
        how many channels to average over. <num_of_channels> needs
        to be divisible by this.

    
    Returns
    -------
        None, the operation is performed in-place.
    '''

    if num_of_channels is None:
        num_of_channels = 100
    if take_average_over is None:
        take_average_over = 5

    for i in range(len(data)):
        max_index = data[i].argmax()
        # average over five value groups (non-rolling)
        averaged_data = data[i][max_index:max_index+num_of_channels].reshape((-1,take_average_over)).mean(axis=1)
        # normalise
        data[i] = averaged_data/averaged_data.max()

def process_output(data):
    '''
    Process output data such that the values are of similar
    order. Data is expected to be component lifetimes on
    even indices, and component intensities on odd indices.
    Scales lifetimes by dividing by 100, and intensities by 
    multiplying by 10.
    '''

    for i in range(len(data)):
        cur_data = data[i]
        cur_data[::2] /= 100
        cur_data[1::2] *= 10
        data[i] = cur_data

def unprocess_output(data):
    '''
    Transform output data back to original form,
    see function process_output for more info.
    '''

    for i in range(len(data)):
        cur_data = data[i]
        cur_data[::2] *= 100
        cur_data[1::2] /= 10
        data[i] = cur_data


def test_fit(regressor, input_vals, output):
    '''
    <output> is true output values.
    '''

    prediction = regressor.predict(input_vals)

    unprocess_output(prediction)
    output = output.copy()
    unprocess_output(output)

    difference = abs(prediction - output)

    format_str = ("{:.3f} "*6).format
    logging.info("")
    logging.info("Comparison of prediction, real and the difference:")
    for i in range(len(prediction)):
        logging.info(format_str(*prediction[i]))
        logging.info(format_str(*output[i]))
        logging.info(format_str(*difference[i]))
        logging.info("")

    logging.info("R2 score: " + str(regressor.score(input_vals, output)))


def main():

    log_folder = os.path.join(
        os.getcwd(),
        "logged_runs"
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

    from read_data import get_train_or_test
    from read_data import get_components

    rng = np.random.default_rng()

    start = time.time()

    folder_path = os.path.join(os.getcwd(), "simdata")
    data_files = os.listdir(folder_path)
    data_files.remove("metadata.txt")

    train_size = 1500
    test_size = 100

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

    # plt.plot(x_train[1])
    # plt.yscale("log")
    # plt.show()


    #TODO: make a pipeline of regressor that also processes
    # input
    take_average_over = 5
    process_input(x_train, take_average_over=take_average_over)
    process_input(x_test, take_average_over=take_average_over)

    logging.info(f"averaging input over {take_average_over} bins")

    # plt.plot(x_train[1])
    # plt.show()

    # print(len(x_train[0]))
    # print(len(x_test))

    # print(x_train[45])

    metadata = read_metadata(folder_path)
    y_train = get_components(metadata, train_files)
    y_test = get_components(metadata, test_files)

    # print(y_train[0])
    process_output(y_train)
    # print(y_train[0])
    process_output(y_test)


    stop = time.time()
    logging.info("time to fetch and process test and train: " + str(stop-start))

    
    # some of these parameters might
    # not be used depending on the solver,
    # but that doesn't really matter (unless
    # the idea is to find good parameters,
    # in which case it is important to not
    # unnecessary change the useless
    # parameters too.)

    # These are just the same parameters
    # as in the article (commented at the
    # start of this file)
    # regressor = MLPRegressor(
    #     hidden_layer_sizes=[150]*7,
    #     activation="relu",
    #     solver="lbfgs",
    #     alpha=0.01,
    #     learning_rate="invscaling",
    #     power_t = 0.5,
    #     max_iter=5e9,
    #     random_state=None,
    #     tol=0.0001,
    #     warm_start=True,
    #     max_fun=15000
    # )

    regressor = MLPRegressor(
        hidden_layer_sizes=[150]*7,
        activation="relu",
        solver="lbfgs",
        alpha=0.01,
        learning_rate="invscaling",
        power_t = 0.5,
        max_iter=5e6,
        random_state=12345,
        tol=0.001,
        warm_start=True,
        max_fun=15000,
    )

    logging.info("\nParameters of regressor:")

    for item in regressor.get_params().items():
        logging.info(item)


    logging.info("\nStarting fitting process at " + time.strftime("%X"))
    fit_start = time.time()
    regressor.fit(x_train,y_train)
    fit_end = time.time()
    logging.info(f"fitting took {fit_end-fit_start:.2f} seconds.")

    # train_prediction = regressor.predict(x_train[0:2])
    # print()
    # print("predictions for train data:")
    # print(train_prediction)
    # print(y_train[0:2])
    # print(train_prediction - y_train[0:2])

    # # not necessarily the best for scoring here?
    # print("R2 score:", regressor.score(x_train, y_train))

    test_with = 10

    logging.info("\n Testing fit with train data:")
    test_fit(regressor, x_train[0:test_with], y_train[0:test_with])

    # test_prediction = regressor.predict(x_test[0:2])
    # print()
    # print("predictions for test data:")
    # print(test_prediction)
    # print(y_test[0:2])
    # print(test_prediction - y_test[0:2])
    # print("R2 score:", regressor.score(x_test, y_test))

    logging.info("\nTesting with test data:")
    test_fit(regressor, x_test[0:test_with], y_test[0:test_with])

    real_folder = os.path.join(
        os.getcwd(),
        "Experimental_data20230215"
    )

    real_data_files = [file for file in os.listdir(real_folder) if file.endswith(".pals")]

    real_metadata = read_metadata(real_folder, "metadata.json")
    real_x = get_train_or_test(real_folder, real_data_files)
    process_input(real_x, take_average_over=take_average_over)
    real_y = get_components(real_metadata, real_data_files)
    process_output(real_y)

    # real_prediction = regressor.predict(real_x)
    # format_str = ("{:.3f} "*6).format
    # for i in range(len(real_prediction)):
    #     print(format_str(*real_prediction[i]))
    #     print(format_str(*real_y[i]))
    #     print()
    
    # print("R2 score:", regressor.score(real_x, real_y))

    logging.info("\nTesting fit with real data")
    test_fit(regressor, real_x, real_y)






    


    # should probably consider background as well, it may be 
    # helpful to somehow encode the fact that the intensities should
    # sum up to 1.




if __name__ == "__main__":
    
    main()