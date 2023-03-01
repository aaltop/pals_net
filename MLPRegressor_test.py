from sklearn.neural_network import MLPRegressor
import numpy as np
import os
import matplotlib.pyplot as plt

# from read_data import read_data
from read_data import read_metadata
import time

# SEE
# Pietrow, Marek & Miaskowski, A.. (2023). 
# Artificial neural network as an effective tool to calculate parameters
# of positron annihilation lifetime spectra. 

def process_input(data, num_of_channels=None):
    '''
    Cut off counts in <data> such that only data from the
    max value onwards up to <num_of_channels> beyond the max 
    is considered, and takes non-rolling means of five values 
    of the result.

    Parameters
    ----------
    data : list of numpy arrays
        The counts for <data.size> simulated spectra.

    num_of_channels : int, divisible by five
        How many channels to take beyond the max channel,
        as in [max_chan:max_chan+num_of_channels]

    
    Returns
    -------
        None, the operation is performed in-place.
    '''

    if num_of_channels is None:
        num_of_channels = 100

    for i in range(len(data)):
        max_index = data[i].argmax()
        # average over five value groups (non-rolling)
        averaged_data = data[i][max_index:max_index+num_of_channels].reshape((-1,5)).mean(axis=1)
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

    

def main():

    print("starting now")

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

    process_input(x_train)
    process_input(x_test)

    # plt.plot(x_train[1])
    # plt.show()

    # print(len(x_train[0]))
    # print(len(x_test))

    print(x_train[45])

    metadata = read_metadata(folder_path)
    y_train = get_components(metadata, train_files)
    y_test = get_components(metadata, test_files)

    print(y_train[0])
    process_output(y_train)
    print(y_train[0])
    process_output(y_test)


    stop = time.time()
    print("time to fetch and process test and train:", stop-start)

    
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

    print("Starting fitting process...")
    fit_start = time.time()
    # Well, it seems to run, but doesn't
    # converge, which isn't terribly 
    # surprising. Need to do some stuff
    # with the data, but at least this works.
    regressor.fit(x_train,y_train)
    fit_end = time.time()
    print(f"fitting took {fit_end-fit_start:.2f} seconds.")
    train_prediction = regressor.predict(x_train[0:2])
    print()
    print("predictions for train data:")
    print(train_prediction)
    print(y_train[0:2])
    print(train_prediction - y_train[0:2])

    # not necessarily the for scoring here?
    print("R2 score:", regressor.score(x_train, y_train))

    test_prediction = regressor.predict(x_test[0:2])
    print()
    print("predictions for test data:")
    print(test_prediction)
    print(y_test[0:2])
    print(test_prediction - y_test[0:2])

    print("R2 score:", regressor.score(x_test, y_test))

    print()
    print("predictions for real data:")
    real_folder = os.path.join(
        os.getcwd(),
        "Experimental_data20230215"
    )

    real_data_files = [file for file in os.listdir(real_folder) if file.endswith(".pals")]

    real_metadata = read_metadata(real_folder, "metadata.json")
    real_x = get_train_or_test(real_folder, real_data_files)
    process_input(real_x)
    real_y = get_components(real_metadata, real_data_files)
    process_output(real_y)

    real_prediction = regressor.predict(real_x)
    format_str = ("{:.3f} "*6).format
    for i in range(len(real_prediction)):
        print(format_str(*real_prediction[i]))
        print(format_str(*real_y[i]))
        print()
    
    print("R2 score:", regressor.score(real_x, real_y))






    


    # should probably consider background as well, it may be 
    # helpful to somehow encode the fact that the intensities should
    # sum up to 1.




if __name__ == "__main__":
    
    main()