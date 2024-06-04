'''
# Instructions:

It should be possible to write simulation data to file by simply
running this file. Depending on the current state of the file,
this will create a new folder in the current working directory,
and write a number of simulation data files in the folder. It is still
recommended to check the file out before doing this.

When wanting to specify exactly how much data to generate, see the main
function in the file. There, change the argument `sims_to_write` of
the `write_many_simulations` function to the preferred number of
simulations to write. In order to change the random simulation parameters,
see the function `random_input_prm`, and change the values there
to the preferred values.

It's recommended to create a separate set of simulated data for training
and tests data and another for validation data.

'''

from simulate_spectra import write_many_simulations
import numpy as np

def random_input_prm(rng):
    '''
    Somewhat randomised input parameters
    for simulation.
    '''


    # 800_000, 1_200_000
    num_events = int(rng.uniform(800_000, 1_200_000))
    # num_events = 5_000_000

    bkg = rng.uniform(0.002, 0.01)#(1_000_000*0.005)/num_events
    # bkg = 0.001
    # remaining "intensity", how much of the num_events are still unspecified
    remaining = 1-bkg

    
    lifetimes = [
        100+rng.uniform(low=-20,high=20),
        350+rng.uniform(low=-70,high=70),
        2000+rng.uniform(low=-200,high=200)
    ]

    
    # TODO: replace with Dirichlet distribution?
    intensities = [0]*3
    # intensities[-1] = rng.uniform(0.001,0.04)*remaining
    intensities[-1] = rng.uniform(0.02,0.06)*remaining
    remaining -= intensities[-1]
    intensities[0] = rng.uniform(0.70, 0.85)*remaining
    intensities[1] = remaining-intensities[0]


    components = list(zip(lifetimes,intensities))

    offset = round(1000 + rng.uniform(low=0, high=100))

    sigma_start = sigma_stop = 65
    input_prm = {"num_events": num_events,
                 "bkg": bkg,
                 "components": components,
                 "bin_size": 25,
                 "time_gate": 15_000,
                 "sigma_start": sigma_start,
                 "sigma_stop": sigma_stop,
                 "offset": offset}
    

    return input_prm


def main():
    import time

    #example input, change input_prm to this to use these as the parameters
    input_prm = {"num_events": 1_000_000,
                "bkg": 0.05,
                "components": [(415, .10), (232, .50), (256, .30), (1200, .05)],
                "bin_size": 25,
                "time_gate": 15_000,
                "sigma_start": 68,
                "sigma_stop": 68,
                "offset": 2000}

    rng = np.random.default_rng()

    # see the function random_input_prm for changing the simulation
    # parameters
    print("Starting to write simulations...")
    start = time.time()
    write_many_simulations(
        sims_to_write=10,
        folder_name="test_simdata",
        input_prm=lambda: random_input_prm(rng),
        repetition_count=1
    )
    stop = time.time()
    print(f"Simulation writing took {stop-start} seconds.")

if __name__ == "__main__":

    main()