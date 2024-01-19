'''

# Instructions:

Everything that's needed to do here (to begin with) is contained in the arguments
of the `main()` function at the bottom. Main things to possibly
change in the function:

- <data_folder>: The folder to fetch evaluation data from.
'''



import torch
import numpy as np
from scipy.stats import kstest
import matplotlib.pyplot as plt
import os

from simulate_spectra import sim_pals
from train import process_input

from read_data import (
    get_simdata,
    get_data_files
)



from pytorch_helpers import r2_score
from pytorchMLP import MLP
from pytorch_helpers import convert_to_tensor

_rng = np.random.default_rng()

def simulate_pred_and_true_spectra(
        pred_and_true_comp,
        input_prm=None
):
    '''
    With the input, makes a prediction using <model>. Then simulates
    a PALS spectrum with both the predicted output and the true
    output, and returns the spectra.

    Parameters
    ----------

    ### pred_and_true_comp : list (length 2) of tensors
        The predicted and true components.

    ### input_prm : dict, default None
        Dictionary of input parameters given to the PALS simulation,
        for example

        input_prm = {"num_events": 1_000_000,
                    "bkg": 0.0,
                    "components": [],
                    "bin_size": 25,
                    "time_gate": 15_000,
                    "sigma_start": 68,
                    "sigma_stop": 68,
                    "offset": 2000}

        components will be replaced by the predicted and given true
        components.

    Returns
    -------

    (pred_bins, pred_hist), (true_bins, true_hist) : tuples of numpy arrays
        The bins and histograms of the spectra.
    '''
    
    if input_prm is None:

        # Actually, it doesn't necessarily matter too much what the other
        # values here are, only the components. Plotting these is 
        # perhaps a little unnecessary in the first place, just 
        # comparing the predicted and true components should really 
        # be enough.
        input_prm = {"num_events": 1_000_000,
                    "bkg": 0.00,
                    "components": [],
                    "bin_size": 25,
                    "time_gate": 15_000,
                    "sigma_start": 68,
                    "sigma_stop": 68,
                    "offset": 2000}
        

    pred, true_comp = pred_and_true_comp

    # format the component list correctly (for the input_prm dict)
    pred_components = [pred[2*i:2*i+2].tolist() for i in range(3)]
    pred_input = input_prm.copy()
    pred_input["components"] = pred_components
    pred_bins, pred_hist = sim_pals(pred_input, _rng)

    true_components = [true_comp[2*i:2*i+2].tolist() for i in range(3)]
    true_input = input_prm.copy()
    true_input["components"] = true_components
    true_bins, true_hist = sim_pals(true_input, _rng)

    return (pred_bins, pred_hist), (true_bins, true_hist)


def test_prediction(
    pred_and_true_comp,
):
    
    '''
    Simulates the spectrum with the components predicted by the model
    and the true components, and compares the simulated spectra.

    Parameters
    ----------

    ### pred_and_true_comp : list (length 2) of tensors
        the predicted and true components.
    '''

    (pred_bins, pred_hist), (true_bins, true_hist) = (
        simulate_pred_and_true_spectra(pred_and_true_comp)
        )

    # here, the components are kind of mean values of the simulated
    # data. Would need to change this obviously, if using some other
    # input parameters for the simulations.
    naive_input = {"num_events": 1_000_000,
                    "bkg": 0.0,
                    "components": [(245, 0.77), (400, 0.2095), (1500, 0.0205)],
                    "bin_size": 25,
                    "time_gate": 15_000,
                    "sigma_start": 68,
                    "sigma_stop": 68,
                    "offset": 2000}
    naive_bins, naive_hist = sim_pals(naive_input, _rng)


    print("\nTwo-sample kstest, model prediction against sim data")
    pred_kstest = kstest(true_hist, pred_hist)
    print(pred_kstest)
    print("Maximum absolute deviation between predicted and true:", np.abs(pred_hist-true_hist).max())
    print(pred_kstest.pvalue)

    print("\nTwo-sample kstest, naive prediction against sim data")
    naive_kstest = kstest(true_hist, naive_hist)
    print(naive_kstest)
    print("Maximum absolute deviation between predicted and true:", np.abs(naive_hist-true_hist).max())


    plt.plot(pred_bins, pred_hist, label="predicted")
    plt.plot(true_bins, true_hist, linestyle="dashed", label="true")
    plt.title("Prediction versus true spectrum, randomly chosen simulation data (validation set)")


    


    # plt.plot(naive_bins, naive_hist, label="naive")

    # plt.plot(np.sort(pred_hist)/pred_hist.max(), label="predicted CDF")
    # plt.plot(np.sort(sim_y_hist)/sim_y_hist.max(), linestyle="dashed", label="true CDF")

    plt.legend()
    plt.xlabel("time (ps)")
    plt.ylabel("counts")
    plt.yscale("log")
    plt.title(f"Predicted vs. true spectrum\n KS test p-value {pred_kstest.pvalue:.4f}")
    plt.show()


    plt.plot(pred_bins, pred_hist-true_hist, label="residual")

    plt.legend()
    plt.xlabel("time (ps)")
    plt.ylabel("counts")
    # plt.yscale("log")
    plt.title(f"Predicted vs. true spectrum residual\n KS test p-value {pred_kstest.pvalue:.4f}")
    plt.show()




def main(
        data_folder,
        data_size=None,
        model_folder=None,
        model_file=None,

):
    '''
    Does the evaluation.


    Parameters
    ----------

    ### data_folder : str
        Folder to fetch the evaluation data from.

    ### data_size : int, default None
        The number of simulation data to use for evaluation. If None,
        uses all data.

    ### model_folder : str, default None
        The name of the folder where the models are saved. Defaults
        to "saved_models".

    ### model_file : str, default None
        The name of the file where the model was saved to. Defaults
        to the last model in <model_folder>.
    '''

        # get simulated data, process
    # --------------------------

    folder_path = os.path.join(os.getcwd(), data_folder)

    # should be fine to specify more than the total amount of files
    # in the folder, but best to keep it to a logical amount
    validation_files, _ = get_data_files(
        data_folder=data_folder, 
        train_size=data_size
    )

    x,y = get_simdata(folder_path,validation_files)

    # Should obviously be the same as when training. Could add a more
    # sensible way to do this, a way to use the same processing as
    # when training.
    take_average_over = 5
    start_index = 0
    num_of_channels = len(x[0])
    process_input(x, num_of_channels, take_average_over=take_average_over, start_index=start_index)
    x = convert_to_tensor(x)
    y = convert_to_tensor(y)

    # ================================================


    # Load up the model data
    # ------------------------------------------------
    if model_folder is None:
        model_folder = "saved_models"

    model_path = os.path.join(os.getcwd(), model_folder)

    if model_file is None:
        model_file = os.listdir(model_path)[-1]

    path_to_saved_model = os.path.join(model_path, model_file)

    with open(path_to_saved_model, "rb") as f:
        state_dict = torch.load(f)

    dev = state_dict["device"]
    dtype = state_dict["dtype"]

    # NOTE: this assumes state_dict contains the model_layers,
    # model_state_dict and the normalisation originally used for
    # the output, as well as used device and data type
    model = MLP(state_dict["model_layers"]).to(dev, dtype)
    model.load_state_dict(state_dict["model_state_dict"])
    model.eval()

    # ================================================



    pred = model(x.to(device=dev, dtype=dtype)).detach()*state_dict["normalisation"]
    pred = pred.to("cpu")


    print("\nValidation set r-squared:")
    separate_r2 = r2_score(pred, y)
    print("Separate:")
    print(separate_r2)
    print("Mean:")
    print(separate_r2.mean())

    # PLot histograms of the distribution of residuals
    # ------------------------------------------------
    residual = y-pred

    residual_normalised = residual/y.mean(dim=0, keepdim=True)

    n_features = residual.size(dim=1)

    fig, axes = plt.subplots(
        nrows = n_features//3 + n_features%3,
        ncols = 3
    )

    for i in range(n_features):

        component_num = i//2 + 1
        r2 = separate_r2[i]

        if i%2 == 0:
            title = f"lifetime {component_num}\n r2 {r2:.2f}"
        else:
            title = f"intensity {component_num}\n r2 {r2:.2f}"

        axis = axes.flatten()[i]
        # axis.hist(y[:,i], label="True")
        # axis.hist(pred[:,i], alpha=0.5, label="Predict")
        axis.hist(residual_normalised[:,i], label="Normalized residual")
        axis.set_title(title)
        axis.legend()
    
    plt.suptitle("Histogram of predicted and true (normalized) component residuals")
    plt.show()
    # ================================================
    


    # choose random components to test
    random_index = _rng.integers(pred.shape[0])
    one_pred = pred[random_index,:]
    one_y = y[random_index,:]

    # print(pred)
    # # so obviously the intensities are not currently constrained to
    # # be anything specifically, as my idea was originally that I'd like
    # # to first see if it can, on its own, get the right values. This
    # # does seem to be close to one and at least not greater, but it
    # # isn't so great to have it less than one either.
    # print(pred[1::2].sum())

    test_prediction([one_pred,one_y])




if __name__ == "__main__":

    # main(
    #     data_folder="simdata_evaluate01",
    #     model_file="model20230503164256.pt"
    # )

    main(
        data_folder="simdata_evaluate02",
    )