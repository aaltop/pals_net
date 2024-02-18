'''

# Instructions:

Everything that's needed to do here (to begin with) is contained in the arguments
of the `main()` function at the bottom. Main things to possibly
change in the function:

- <data_folder>: The folder to fetch evaluation data from.
'''



import torch
import numpy as np
from scipy.stats import (
    kstest,
    ttest_ind
)
import matplotlib.pyplot as plt
import os

from simulate_spectra import sim_pals

from read_data import (
    get_simdata,
    get_data_files
)

from processing import (
    process_input
)

from helpers import one_line_print


from pytorch_helpers import r2_score
from models import MLP, NeuralNet
from pytorch_helpers import convert_to_tensor

_rng = np.random.default_rng()

def simulate_pred_and_true_spectra(
        pred_and_true_comp,
        input_prm=None
):
    '''
    Simulates a PALS spectrum with both the predicted output and the true
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
    rng
):
    
    '''
    Simulates the spectrum with the components predicted by the model
    and the true components, and compares the simulated spectra.

    Parameters
    ----------

    ### pred_and_true_comp : list of lists (of length 2) of tensors
        the predicted and true components.

    ### rng : numpy rng
    '''


    # Calculate KS metrics for each evaluate data point
    # ------------------------------
    ks_metrics = []
    p_thres = 0.05
    rejects = 0
    # The shuffle doesn't really do anything meaningful at this
    # point, it just ensures that some variables further down
    # need not be changed. Could probably just pick a random
    # set of hists from here as a more sensible alternative.
    rand_idx = np.arange(len(pred_and_true_comp))
    rng.shuffle(rand_idx)
    rand_idx = [rand_idx[0]]
    for step,i in enumerate(rand_idx):
        one_line_print(f"Simulating spectra {step+1}/{len(rand_idx)}")
        comps = pred_and_true_comp[i]
        (pred_bins, pred_hist), (true_bins, true_hist) = (
            simulate_pred_and_true_spectra(comps)
            )
        pvalue = kstest(true_hist, pred_hist).pvalue
        rejects += 1 if pvalue < p_thres else 0
        ks_metrics.append(kstest(true_hist, pred_hist).pvalue)

    # ====================================

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

    # Calculate the intervals (mean and variance) for one predicted and true case
    # -----------------------------------
    n = 100
    preds = [0]*n
    trues = [0]*n
    for i in range(n):

        (pred_bins, pred_hist), (true_bins, true_hist) = (
            simulate_pred_and_true_spectra(comps)
            )
        
        preds[i] = pred_hist
        trues[i] = true_hist

    preds = np.vstack(preds)
    preds_std = np.std(preds, axis=0)
    preds_mean = np.mean(preds, axis=0)
    preds_lower = preds_mean-2*preds_std
    preds_upper = preds_mean+2*preds_std
    preds_ci = [preds_lower, preds_mean, preds_upper]

    trues = np.vstack(trues)
    trues_std = np.std(trues, axis=0)
    trues_mean = np.mean(trues, axis=0)
    trues_lower = trues_mean-2*trues_std
    trues_upper = trues_mean+2*trues_std
    trues_ci = [trues_lower, trues_mean, trues_upper]


    # ===========================================



    plt.plot(pred_bins, np.vstack(preds_ci).T, label="predicted")
    plt.plot(true_bins, np.vstack(trues_ci).T, linestyle="dashed", label="true")
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




    plt.plot(pred_bins, np.vstack((trues_ci[0]-trues_ci[1],preds_ci[1]-trues_ci[1], trues_ci[2]-trues_ci[1])).T, label="residual")

    # Two-sample t-test p-values for n simulated spectra one test for each bin
    ttest_p = ttest_ind(trues, preds, equal_var=False).pvalue
    p_ratio = np.sum(ttest_p >= 0.05)/len(ttest_p)

    plt.legend()
    plt.xlabel("time (ps)")
    plt.ylabel("counts")
    # plt.yscale("log")
    plt.title(f"Predicted vs. true spectrum residual, normalised\n KS test p-value {pred_kstest.pvalue:.4f} \n two-sample t-test ratio of >=0.05: {p_ratio:.4f}")
    plt.show()

    # plt.hist(ttest_ind(trues, preds, equal_var=False).pvalue, label="var False")
    # plt.hist(ttest_ind(trues, preds, equal_var=True).pvalue, label="var True", alpha=0.5)
    # plt.title(f"Two-sample t-test p-values for {n} simulated spectra\n one test for each bin")
    # plt.show()

    plt.hist(ks_metrics)
    plt.title(f"Kolmogorov-Smirnov test p-values histogram, rejected: {rejects}")
    plt.show()




def main(
        data_folder,
        data_size=None,
        model_folder=None,
        model_file=None,
        verbose=False

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

    ### verbose : boolean, default False
        Whether to have a verbose output. Verbose output includes the
        corresponding log file when possible.
    '''

    # Load up the model data
    # ------------------------------------------------
    if model_folder is None:
        model_folder = "saved_models"

    model_path = os.path.join(os.getcwd(), model_folder)

    if model_file is None:
        model_file = os.listdir(model_path)[-1]

    path_to_saved_model = os.path.join(model_path, model_file)

    with open(path_to_saved_model, "rb") as f:
        train_dict = torch.load(f)


    log_file_path = train_dict.get('log_file_path', None)
    if verbose and not (log_file_path is None):

        print("---------------------------------------")
        print(f"Log file at {log_file_path}:\n")
        log_str = "-LOG"*10+"-"
        print(log_str)
        print()
        with open(log_file_path, "r", encoding="utf-8") as f:
            print(f.read())
        
        print(log_str)
    
    # ==================================================

    # get simulated data, process
    # --------------------------

    folder_path = os.path.join(os.getcwd(), data_folder)

    # should be fine to specify more than the total amount of files
    # in the folder, but best to keep it to a logical amount
    validation_files, _ = get_data_files(
        folder_path, 
        train_size=data_size
    )

    x,y = get_simdata(folder_path,validation_files)

    # this first is for older train_dicts, which did not have the
    # processing arguments saved
    if not ("process_input_parameters" in train_dict):
        take_average_over = 5
        start_index = 0
        num_of_channels = len(x[0])
        process_input(x, num_of_channels, take_average_over, start_index)
    elif not (train_dict["process_input_parameters"] is None):
        process_input(x, **train_dict["process_input_parameters"])

    x = convert_to_tensor(x)
    x /= torch.amax(x, dim=1).reshape((-1,1))
    y = convert_to_tensor(y)

    # ================================================


    
    # Ready the model
    # ------------------------------------------------

    # older train_dicts didn't have these, so use get with the default
    # specified
    dev = train_dict.get("device", "cpu")
    dtype = train_dict.get("dtype", torch.float32)


    # NOTE: this assumes state_dict contains the model_layers,
    # model_state_dict and the normalisation originally used for
    # the output, as well as used device and data type

    network = NeuralNet
    # for older train_dict contents
    if "model_layers" in train_dict:
        model = MLP(train_dict["model_layers"]).to(dev, dtype)
    else:
        model = network(**train_dict["model_kwargs"]).to(dev, dtype)

    model.load_state_dict(train_dict["model_state_dict"])
    model.eval()

    # ================================================



    pred = model(x.to(device=dev, dtype=dtype)).detach()*train_dict["normalisation"]
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

    # number of predicted values
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
    # TODO: add softmax to intensities
    # print(pred[1::2].sum())

    # test_prediction([one_pred,one_y])
    test_prediction([(pred[i,:], y[i,:]) for i in range(len(pred))], _rng)




if __name__ == "__main__":

    main(
        data_folder="simdata_evaluate01",
        verbose=False
    )

    # main(
    #     data_folder="simdata_evaluate02",
    #     verbose=True
    # )