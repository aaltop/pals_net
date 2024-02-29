import os
import contextlib

import json
import numpy as np





# TODO: Might be best to just make a PyTorch dataloader for the
# PyTorch stuff at least.

# NOTE: Currently it's best to use get_simdata if just reading in
# the data and components

def get_data_files(
        folder_path:str,
        train_size:int=None,
        test_size:int=None,
        rng:np.random.Generator=None,
        permute_data=False
    ):

    '''
    Gets the data files (names of data files) in <data_folder>,
    optionally does permutation to shuffle train and test.

    Parameters
    ----------


    ### train_size, test_size : int, default None
        Number of files to use as train and test data source. If <test_size>
        is None, the <train_size> files will be returned. If both are None,
        all the files are returned. Cannot have just <train_size> be None,
        mainly to keep the interface somewhat sensible.

    ### permute_data : Boolean, default False
        Whether to permute the data files. If both <train_size> and
        <test_size> are None, no permutation is performed
        as all files are returned regardless.


    Returns
    -------

    ### train_files, test_files : list of strings
        Names of the train and test files in <data_folder>. If
        <test_files> or both are None, returns <test_files> as empty list.
    '''


    data_files = os.listdir(folder_path)

    for to_remove in ("metadata.txt", "metadata.json"):
        if to_remove in data_files:
            data_files.remove(to_remove)

    if train_size is None:
        if not (test_size is None):
            # just to keep some sense in the use
            raise ValueError("Cannot have <train_size> but not <test_size> be None")
        
        # no point in permutation, really.
        return data_files, []

    if permute_data:
        if rng is None:
            rng = np.random.default_rng()

        data_files = rng.permutation(data_files)

    train_files = data_files[:train_size]
    if test_size is None:
        return train_files, []

    test_files = data_files[train_size:train_size+test_size]
    return train_files, test_files

def read_metadata(folder_path, metadata_name=None) -> dict:

    if metadata_name is None:
        metadata_name = "metadata.txt"

    metadata_path = os.path.join(folder_path, metadata_name)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    
    return metadata

def real_meta_to_json(file_path):
    '''
    Get real metadata (metadata that came with the real experiment data)
    as proper json.
    '''

    with open(file_path, "r", encoding="utf-8") as f:

        lines = f.readlines()

    # remove empty lines
    while True:
        try:
            lines.remove("\n")
        except ValueError:
            break

    # wrangle the strings into suitable format
    lines = [line.rstrip() for line in lines]
    lines = [line.replace("(","[") for line in lines]
    lines = [line.replace(")","]") for line in lines]
    lines = [line.replace(" .", "0.") for line in lines]
    
    
    file_name_lines = [i for i,name in enumerate(lines) if name.endswith(".pals")]
    metadata = {}

    # print(lines)

    # pick the metadata from in between the file names,
    # parse to json and set into dict.
    for index in range(len(file_name_lines)):

        start = file_name_lines[index]+1
        file_name = lines[file_name_lines[index]]
        if index != len(file_name_lines)-1:
            stop = file_name_lines[index+1]
            metadata[file_name] = json.loads("".join(lines[start:stop]))
        # for the last file, just take the rest of the lines
        else:
            metadata[file_name] = json.loads("".join(lines[start:]))


    return metadata


def write_meta(metadata, file_path):
    '''
    Write json-compliant <metadata> to file <file_path>.
    '''

    # perhaps a tad unnecessary altogether.
    with open(file_path, "w", encoding="utf-8") as f:

        json.dump(metadata, f, indent="\t")
    



def get_components(metadata:dict, file_names:list, return_vals="all") -> list:
    '''
    Get lifetime and intensity components for given
    data determined by <file_names>, a list of data filename strings. 
    <metadata> can be fetched with the use of the function read_metadata.

    Returns components in a list as flattened numpy arrays.

    Parameters
    ----------

    metadata : dict
        dictionary of simulated data metadata, which has a "components"
        part, which consists of pairs of lifetimes and relative
        intensities
    
    file_names : list
        list of file names - simulations - to read metadata for
    
    return_vals : string
        one of "all", "lifetimes" or and "intensities"
    '''

    all_comps = [0]*len(file_names)

        
    match return_vals:
        case "all":
            for i,file_name in enumerate(file_names):
                all_comps[i] = np.array(
                metadata[file_name]["components"]).flatten()
        case "lifetimes":
            for i,file_name in enumerate(file_names):
                all_comps[i] = np.array(
                metadata[file_name]["components"]).flatten()[::2]
        case "intensities":
            for i,file_name in enumerate(file_names):
                all_comps[i] = np.array(
                metadata[file_name]["components"]).flatten()[1::2]
        case _:
            raise ValueError("<return_vals> must be one of 'all', 'lifetimes' or 'intensities'")

    return all_comps

def read_data(folder_path, file_name) -> np.ndarray:


    file_path = os.path.join(folder_path, file_name)
    with open(file_path, "r", encoding="utf-8") as f:
        f.readline() # skip the metadata
        data = np.loadtxt(f, dtype="float64")


    return data

def get_train_or_test(folder_path:str, file_names, col_names=None):
    '''
    Returns the counts for the data in the files <file_names>
    from the folder <folder_path>.


    Parameters
    ----------

    ### folder_path : string
        path of the folder containing the data
    
    ### file_names : list of strings
        names of the data files to get counts from

    ### col_names : tuple of strings
        the names of the columns of the data. If None, defaults
        to ("time_ps", "counts")

    
    
    Returns
    -------

    ### data : list of numpy arrays of counts
    '''

    if col_names is None:
        col_names = ("time_ps", "counts")
    
    data = [0]*len(file_names)

    for i,file_name in enumerate(file_names):
        data[i] = read_data(folder_path, file_name)[:,1]

    return data

def get_simdata(folder_path:str, file_names:list[str], inputs):
    '''
    Returns both the simulation counts and the inputs used for the
    simulation, both of which should be in one file. The inputs
    (simulation input parameters) should be on the first line of the
    file as a (json deserialisable) dictionary, and the simulation
    time and counts make up the rest of the file, in two columns.

    Parameters
    ----------

    ### folder_path : string
        The path of the folder the data files are in.


    ### file_names : list of strings
        The names of the files to get the data from.

    ### inputs : list of strings
        The used inputs. The strings should match the keys of the JSON
        dictionary. The items are expected to be either scalar or
        a list, with any lists flattened in to the output. Lists should
        contain lists as elements, with the inner lists containing scalars.

    Returns
    -------

    ### counts: list of numpy arrays
        The counts of the spectra. Currently needs to be a list because
        may get processed later, which requires the list.

    ### components : numpy array
        The inputs used in simulating the spectra, in the order specified
        by <inputs>.
    '''

    components = [0]*len(file_names)
    counts = [0]*len(file_names)


    # change to data folder so don't have to create the path
    # strings for every data file
    # not thread-safe
    with contextlib.chdir(folder_path):

        for i,file in enumerate(file_names):
            with open(file, "r", encoding="utf-8") as f:

                # get the components (lifetimes and intensities)
                input_prm_str = f.readline()
                input_prm = json.loads(input_prm_str)
                comps = []
                for key in inputs:
                    
                    _input = input_prm[key]
                    # The assumption is that the "components" are in
                    # a list of lists, with inner lists having
                    # lifetime, intensity pairs, while other input
                    # parameters are scalar.
                    # This is already unnecessary complex to work with,
                    # so it's better to limit it to a very specific case,
                    # rather than give more wiggle-room for stupid stuff.
                    # It would be better to have the input parameters
                    # in a nicer format in the saved files, where every
                    # item is scalar, but I'd rather not have to 
                    # generate all that data again right now, and it's
                    # generally a hassle. Of course,
                    # the assumption would be that the simulation would
                    # not really change all that much, though it certainly
                    # could.
                    if isinstance(_input, list):
                        for sub_list in _input:
                            comps += sub_list
                    else:
                        comps.append(_input)
                    
                components[i] = comps

                # get the counts
                count_col = np.loadtxt(f, dtype="float64", usecols=1)
                counts[i] = count_col

        return counts, np.array(components)


def get_input_prm(folder_path, file_names:list[str]):
    '''
    Get the input parameters used to create the data in <folder_path>.
    '''

    with contextlib.chdir(folder_path):

        for i,file in enumerate(file_names):
            with open(file, "r", encoding="utf-8") as f:
                if 0 == i:
                    input_prm = json.loads(f.readline())
                    for key in input_prm:
                        input_prm[key] = [input_prm[key]]
                else:
                    dicti = json.loads(f.readline())
                    for key in dicti:
                        input_prm[key].append(dicti[key])

    return input_prm
  

def test_metadata():
    folder = os.path.join(os.getcwd(), "simdata")
    metadata = read_metadata(folder)
    print(metadata["simdata_00231.pals"])

def test_data():
    import matplotlib.pyplot as plt

    data = read_data("simdata_train01","simdata_00999.pals")
    plt.plot(data[:,0], data[:,1])
    plt.yscale("log")
    plt.show()

def test_meta_to_json():

    folder_path = os.path.join(
        os.getcwd(), 
        "Experimental_data20230215")
    
    file_path = os.path.join(folder_path, "metadata.txt")
    data_files = os.listdir(folder_path)
    data_files = [file for file in data_files if file.endswith(".pals")]

    metadata = real_meta_to_json(file_path)

    for file_name in data_files:
        print(metadata[file_name])

    # write_meta(metadata, os.path.join(folder_path, "metadata.json"))


def test_read_real():

    real_folder =  os.path.join(os.getcwd(),"Experimental_data20230215")

    real_data_files = [
        file for file in os.listdir(real_folder) if file.endswith(".pals")
    ]


    real_meta = read_metadata(
        real_folder, 
        "metadata.json")
    
    print(get_components(real_meta, real_data_files))
    
    print(get_train_or_test(real_folder, real_data_files))

    # print(read_data(real_folder, "data_0005.pals"))

def test_get_simdata():

    import time

    start = time.time()

    data_folder = "temp_file"

    folder_path = os.path.join(
        os.getcwd(),
        data_folder
    )
    train_files, test_files = get_data_files(
        folder_path,
        )
    
    inputs = (
        "components",
        "bkg",
    )
    train_counts, train_components = get_simdata(folder_path, train_files, inputs)
    # test_counts, test_components = get_simdata(folder_path, test_files)

    print(f"Took {time.time()-start} seconds.")

    print(train_components.shape)
    print(len(train_counts))
    print(train_components)

def test_get_input_prm():

    folder_path = os.path.join(
        os.getcwd(),
        "simdata_train01"
    )

    get_data_files(folder_path)
    input_prm = get_input_prm(folder_path, get_data_files(folder_path)[0])

    for key,value in input_prm.items():
        input_prm[key] = [np.min(value, axis=0), np.max(value, axis=0)]
    
    print(input_prm)

def main():

    folder_path = os.path.join(os.getcwd(), "simdata_train01")
    file_names = os.listdir(folder_path)
    # should only contain simdata files
    file_names.remove("metadata.txt")
    train = get_train_or_test(folder_path, file_names[:100])
    
    first = train[0]
    max_val_index = first.argmax()
    print(max_val_index)
    print(first[max_val_index:])

if __name__ == "__main__":

    test_data()
    main()