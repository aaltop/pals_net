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

def process_input(data, num_of_channels=None, take_average_over=None, start_index=None):
    '''
    Cut off counts in <data> such that only data from the
    max value onwards up to <num_of_channels> beyond the max 
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
        averaged_data = data[i][start_index:start_index+num_of_channels].reshape((-1,take_average_over)).mean(axis=1)
        # normalise
        data[i] = averaged_data/averaged_data.max()

def process_output(data, output_type = "all"):
    '''
    Process output data such that the values are of similar
    order. Data is expected to be component lifetimes on
    even indices, and component intensities on odd indices, if
    <output_type> is "all". Scales lifetimes by dividing by 100, 
    and intensities by multiplying by 10.

    output_type : string
        one of "all", "lifetimes" and "intensities".
    '''

    match output_type:
        case "all":
            for i in range(len(data)):
                cur_data = data[i]
                cur_data[::2] /= 100
                cur_data[1::2] *= 10
                data[i] = cur_data

        case "lifetimes":
            for i in range(len(data)):
                data[i] /= 100
        case "intensities":
            for i in range(len(data)):
                data[i] *= 10
        case _:
            raise ValueError("<return_vals> must be one of 'all', 'lifetimes' and 'intensities'")

def unprocess_output(data, output_type="all"):
    '''
    Transform output data back to original form,
    see function process_output for more info.

    output_type : string
        one of "all", "lifetimes" and "intensities".
    '''

    match output_type:
        case "all":
            for i in range(len(data)):
                cur_data = data[i]
                cur_data[::2] *= 100
                cur_data[1::2] /= 10
                data[i] = cur_data

        case "lifetimes":
            for i in range(len(data)):
                data[i] *= 100
        case "intensities":
            for i in range(len(data)):
                data[i] /= 10
        case _:
            raise ValueError("<return_vals> must be one of 'all', 'lifetimes' and 'intensities'")


def test_fit(regressor, input_vals, output, output_type="all"):
    '''
    <output> is true output values.

    <output_type> is one of "all", "lifetimes" and "intensities".

    <input_vals> should be in processed form, <output> in non-processed
    from (need to change this)
    '''

    prediction = regressor.predict(input_vals)

    unprocess_output(prediction, output_type)


    difference = abs(prediction - output)

    output_len = len(output[0])
    format_str = ("{:.3f} "*output_len).format
    logging.info("")
    logging.info("Comparison of prediction, real and the difference:")
    for i in range(len(prediction)):
        logging.info(format_str(*prediction[i]))
        logging.info(format_str(*output[i]))
        logging.info(format_str(*difference[i]))
        logging.info("")

    logging.info("true output mean:")
    logging.info(format_str(*np.array(output).mean(axis=0)))

    logging.info("R2 score: " + str(regressor.score(input_vals, output)))


def main():

    from sklearn.neural_network import MLPRegressor

    # setup logger
    # -----------------
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
    # =========================

    from read_data import get_train_or_test
    from read_data import get_components

    rng = np.random.default_rng()

    start = time.time()

    # get training and test input, process them
    # --------------------------

    folder = "simdata_more_random"
    logging.info(f"simdata folder: {folder}")
    folder_path = os.path.join(os.getcwd(), folder)
    data_files = os.listdir(folder_path)
    data_files.remove("metadata.txt")

    train_size = 50
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

    # plt.plot(x_train[1])
    # plt.yscale("log")
    # plt.show()



    #TODO: make a pipeline of regressor that also processes input
    num_of_channels = 100
    take_average_over = 5
    process_input(x_train, num_of_channels, take_average_over=take_average_over)
    process_input(x_test, num_of_channels, take_average_over=take_average_over)

    logging.info(f"averaging input over {take_average_over} bins")

    # ====================================================

    # plt.plot(x_train[1])
    # plt.show()

    # print(len(x_train[0]))
    # print(len(x_test))

    # print(x_train[45])

    # get training and test output, process
    # -------------------------------------

    # what to fit on: lifetimes, intensities or both ("all")
    output_to_fit = "lifetimes"

    metadata = read_metadata(folder_path)
    y_train = get_components(metadata, train_files, output_to_fit)
    y_test = get_components(metadata, test_files, output_to_fit)

    process_output(y_train, output_to_fit)


    # =======================================

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

    max_iter = int(5e6)
    hidden_layers = np.array([150]*7)- 20*np.arange(0,7)
    regressor = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="lbfgs",
        alpha=0.001,
        learning_rate="invscaling",
        power_t = 0.5,
        max_iter=max_iter,
        random_state=12345,
        tol=1e-4,
        warm_start=True,
        max_fun=25000,
    )

    logging.info("\nParameters of regressor:")

    for item in regressor.get_params().items():
        logging.info(item)


    logging.info("\nStarting fitting process at " + time.strftime("%X"))
    fit_start = time.time()
    regressor.fit(x_train,y_train)
    fit_end = time.time()
    logging.info(f"fitting took {fit_end-fit_start:.2f} seconds.")

    logging.info("\nnumber of iterations:")
    logging.info(regressor.n_iter_)

    logging.info("\nType of score function:")
    logging.info(regressor.loss)
    logging.info("Loss value at last iteration:")
    logging.info(regressor.loss_)

    unprocess_output(y_train, output_to_fit)
    

    test_with = min(10, train_size, test_size)
    logging.info("\nTesting fit with train data:")
    test_fit(regressor, x_train[0:test_with], y_train[0:test_with], output_to_fit)
    logging.info("\nTesting with test data:")
    test_fit(regressor, x_test[0:test_with], y_test[0:test_with], output_to_fit)

    real_folder = os.path.join(
        os.getcwd(),
        "Experimental_data20230215"
    )

    real_data_files = [file for file in os.listdir(real_folder) if file.endswith(".pals")]

    real_metadata = read_metadata(real_folder, "metadata.json")
    real_x = get_train_or_test(real_folder, real_data_files)
    process_input(real_x, num_of_channels, take_average_over=take_average_over)
    real_y = get_components(real_metadata, real_data_files, output_to_fit)

    # real_prediction = regressor.predict(real_x)
    # format_str = ("{:.3f} "*6).format
    # for i in range(len(real_prediction)):
    #     print(format_str(*real_prediction[i]))
    #     print(format_str(*real_y[i]))
    #     print()
    
    # print("R2 score:", regressor.score(real_x, real_y))

    logging.info("\nTesting fit with real data")
    test_fit(regressor, real_x, real_y, output_to_fit)


    


    # should probably consider background as well, it may be 
    # helpful to somehow encode the fact that the intensities should
    # sum up to 1.




if __name__ == "__main__":
    
    main()