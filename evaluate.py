'''

# Instructions:

Everything that's needed to do here (to begin with) is contained in the arguments
of the `main()` function at the bottom. Main things to possibly
change in the function:

- <data_folder>: The folder to fetch evaluation data from.
'''

import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import numpy as np
from scipy.stats import (
    kstest,
    ttest_ind
)
import matplotlib.pyplot as plt

from simulate_spectra import sim_pals

from read_data import (
    get_simdata,
    get_data_files
)

from processing import (
    process_input_mlp,
    process_input01
)

from helpers import one_line_print

from plotting_utility import PlotSaver


from pytorch_helpers import r2_score, convert_to_tensor

from models import MLP, NeuralNet, PALS_MSE, PALS_GNLL
import load_model

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
    # rand_idx = [rand_idx[0]]
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


def evaluate_model(
    model,
    train_dict,
    x,
    y
):


    dev = train_dict.get("device", "cpu")
    dtype = train_dict.get("dtype", torch.float32)

    # ================================================

    pred = model(x.to(device=dev, dtype=dtype))
    # if output prediction comes as multiple tensors in tuple. Map columns in
    # the prediction tensors to how they are in the original, true output "y"
    if "idx" in train_dict:

        idx_obj = train_dict["idx"]
        if isinstance(idx_obj, dict):
            idx_obj = tuple(idx_obj.values())

        col_index = []
        for idx_list in idx_obj:
            col_index += idx_list

        # sort into correct index order
        pred_to_y_conversion = [val[0] for val in sorted(enumerate(col_index), key=lambda x: x[1])]
        # use sorted indices to get the correct order
        pred = torch.column_stack(pred)[:, pred_to_y_conversion]


    if "normalisation" in train_dict:
        pred = pred.detach()*train_dict["normalisation"]
    else:
        proc, kwargs = train_dict["output_normalisation"]
        output_processing = proc(**kwargs)
        pred = output_processing.inv_process(pred.detach())

    pred = pred.to("cpu")



    print("\nValidation set r-squared:")
    separate_r2 = r2_score(pred, y)
    print("Separate:")
    print(separate_r2)
    print("Mean:")
    print(separate_r2.mean())

    return pred, separate_r2


def evaluate_gnll_model(
    model,
    train_dict,
    x,
    y,
    comp_names,
    plot_saver:PlotSaver
):


    dev = train_dict.get("device", "cpu")
    dtype = train_dict.get("dtype", torch.float32)

    # ================================================

    pred = model(x.to(device=dev, dtype=dtype))

    (normal,normal_var),(softmax, softmax_var) = pred
    # if output prediction comes as multiple tensors in tuple. Map columns in
    # the prediction tensors to how they are in the original, true output "y"
    if "idx" in train_dict:

        idx_obj = train_dict["idx"]
        if isinstance(idx_obj, dict):
            idx_obj = tuple(idx_obj.values())

        col_index = []
        for idx_list in idx_obj:
            col_index += idx_list

        # sort into correct index order
        pred_to_y_conversion = [val[0] for val in sorted(enumerate(col_index), key=lambda x: x[1])]
        # use sorted indices to get the correct order
        # print(pred_to_y_conversion)
        # print(torch.column_stack((normal, softmax)).size())
        pred = torch.column_stack((normal, softmax))[:, pred_to_y_conversion]
        pred_var = torch.column_stack((normal_var, softmax_var))[:, pred_to_y_conversion]


    if "normalisation" in train_dict:
        pred = pred.detach()*train_dict["normalisation"]
        

        # To get actual variance, need to multiply by the square of 
        # the multiplying normalisation constant
        pred_var = pred_var.detach()*(train_dict["normalisation"]**2)
    else:
        proc, kwargs = train_dict["output_normalisation"]
        output_processing = proc(**kwargs)

        pred = output_processing.inv_process(pred.detach())
        pred_var = output_processing.inv_process_var(pred_var.detach())

    pred = pred.to("cpu")
    pred_var = pred_var.to("cpu")


    print("\nValidation set r-squared:")
    separate_r2 = r2_score(pred, y)
    print("Separate:")
    print(separate_r2)
    print("Mean:")
    print(separate_r2.mean())

    # plot each of the true values and the predicted ones,
    # together with 2*std confidence intervals. Sort in ascending order
    # by the true values
    for i,name in enumerate(comp_names):
        true, true_sort = torch.sort(y[:,i])
        predicted = pred[:, i][true_sort]
        std2 = (2*pred_var[:,i]**0.5)[true_sort]

        num_val = np.arange(len(std2))

        fig = plt.figure(figsize=(19.2, 10.8))
        plt.plot(predicted, label="predicted")
        plt.plot(true, label="true")
        plt.fill_between(num_val, predicted-std2, predicted+std2, alpha=0.5, label="2*std prediction interval")
        plt.legend()
        plt.title(f"True and predicted values for {name}\nMean of two standard deviations: {torch.mean(std2).item():.3f}")
        plt.xlabel("number of data file")
        plt.ylabel("Value of component")
        plot_saver.save(name, fig)
        plt.close(fig)

    return (pred, pred_var), separate_r2

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

    train_dict = load_model.load_train_dict(model_folder, model_file)


    log_file_path = train_dict.get('log_file_path', None)
    if verbose and not (log_file_path is None):

        try:
            with open(log_file_path, "r", encoding="utf-8") as f:
                print("---------------------------------------")
                print(f"Log file at {log_file_path}:\n")
                log_str = "-LOG"*10+"-"
                print(log_str)
                print()
                print(f.read())
            
            print(log_str)
        except FileNotFoundError:
            print("Log file not found; it's name may have changed.")
    
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

    inputs = train_dict.get("sim_inputs", ("components"))
    x, y, comp_names = get_simdata(folder_path,validation_files, inputs)

    # this first is for older train_dicts, which did not have the
    # processing arguments saved
    if not ("process_input_parameters" in train_dict):
        take_average_over = 5
        start_index = 0
        num_of_channels = len(x[0])
        process_input_mlp(x, num_of_channels, take_average_over, start_index)
    elif (train_dict["process_input_parameters"]["func_name"] == "process_input_mlp"):
        process_input_mlp(x, **train_dict["process_input_parameters"])

    x = convert_to_tensor(x)
    x = process_input01(x)
    y = convert_to_tensor(y)

    # ================================================


    
    # Ready the model
    # ------------------------------------------------

    plot_saver = PlotSaver(data_folder)
    
    model_class = train_dict.get("model_class", PALS_GNLL)
    model = load_model.load_network(train_dict, model_class)
    model.eval()
    if PALS_MSE is model_class:
        evaluate_model(model, train_dict, x, y)
    elif PALS_GNLL is model_class:
        (pred, pred_var), separate_r2 = evaluate_gnll_model(model, train_dict, x, y, comp_names, plot_saver)

    # PLot histograms of the distribution of residuals
    # ------------------------------------------------
    residual = y-pred

    true_means = y.mean(dim=0)
    residual_normalised = residual/y

    axes_mosaic = [
        ["lifetime 1", "intensity 1", "lifetime 2", "intensity 2"],
        ["lifetime 3", "intensity 3", "background", "background"]
    ]

    name_to_idx = {
        "lifetime 1":0,
        "intensity 1":1,
        "lifetime 2":2,
        "intensity 2":3,
        "lifetime 3":4,
        "intensity 3":5,
        "background":6
    }

    fig, ax_dict = plt.subplot_mosaic(axes_mosaic)

    # fig, axes = plt.subplots(
    #     nrows = n_features//3 + n_features%3,
    #     ncols = 3
    # )


    for ax_name, axis in ax_dict.items():

        idx = name_to_idx[ax_name]
        r2 = separate_r2[idx]


        title = f"{ax_name}, true mean: {true_means[idx]:.4f}\n r^2 {r2:.2f}"


        # axis.hist(y[:,i], label="True")
        # axis.hist(pred[:,i], alpha=0.5, label="Predict")
        axis.hist(residual_normalised[:,idx])
        axis.set_title(title)
        # axis.legend()
    
    plt.suptitle("Histogram of predicted and true (normalized) component residuals")
    plot_saver.save("r2_histograms", fig)
    plt.show()
    # ================================================
    


    # choose random components to test
    random_index = _rng.integers(pred.shape[0])
    one_pred = pred[random_index,:]
    one_y = y[random_index,:]


    raise SystemExit

    # test_prediction([one_pred,one_y])
    test_prediction([(pred[i,:][:-1], y[i,:][:-1]) for i in range(len(pred))], _rng)




if __name__ == "__main__":

    # main(
    #     data_folder="simdata_evaluate01",
    #     # model_file="model20240307171529.pt",
    #     verbose=False
    # )

    # main(
    #     data_folder="simdata_evaluate03",
    #     model_file = "model20240312170320.pt",
    #     verbose=False
    # )

    # main(
    #     data_folder="simdata_evaluate05",
    #     verbose=False
    # )

    # main(
    #     data_folder="simdata_evaluate06",
    #     verbose=False
    # )

    # main(
    #     data_folder="simdata_evaluate07",
    #     verbose=False
    # )

    main(
        data_folder="simdata_evaluate10",
        verbose=False
    )