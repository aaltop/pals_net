from sklearn.neural_network import MLPRegressor
import numpy as np
import os

from read_data import read_data
from read_data import read_metadata

# SEE
# Pietrow, Marek & Miaskowski, A.. (2023). 
# Artificial neural network as an effective tool to calculate parameters
# of positron annihilation lifetime spectra. 



def main():

    from read_data import get_train_or_test
    from read_data import get_components

    rng = np.random.default_rng()

    folder_path = os.path.join(os.getcwd(), "simdata")
    data_files = os.listdir(folder_path)
    data_files.remove("metadata.txt")

    perm_data_files = rng.permutation(data_files)

    train_size = 100
    train_files = perm_data_files[:train_size]
    test_size = 100
    test_files = perm_data_files[train_size:train_size+test_size]

    x_train = get_train_or_test(folder_path, train_files)
    x_test = get_train_or_test(folder_path, test_files)

    print(len(x_train[0]))
    print(len(x_test[0]))

    metadata = read_metadata(folder_path)
    y_train = get_components(metadata, train_files)
    y_test = get_components(metadata, test_files)

    
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
    regressor = MLPRegressor(
        hidden_layer_sizes=[150]*7,
        activation="relu",
        solver="lbfgs",
        alpha=0.01,
        learning_rate="invscaling",
        power_t = 0.5,
        max_iter=5e9,
        random_state=None,
        tol=0.0001,
        warm_start=True,
        max_fun=15000
    )

    print("Starting fitting process...")
    
    # Well, it seems to run, but doesn't
    # converge, which isn't terribly 
    # surprising. Need to do some stuff
    # with the data, but at least this works.
    regressor.fit(x_train,y_train)
    print(regressor.score(x_train, y_train))




if __name__ == "__main__":

    main()