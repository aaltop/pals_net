import json
import pandas as pd
import os
import numpy as np

def get_data_files(
        data_folder,
        train_size,
        test_size,
        rng,
        permute_data=False
    ):

    '''
    Gets the data files (names of data files) in <data_folder>,
    optionally does permutation to shuffle train and test.

    Returns
    -------

    ### train_files, test_files : list of strings
        Names of the train and test files in <data_folder>.
    '''

    folder_path = os.path.join(os.getcwd(), data_folder)
    data_files = os.listdir(folder_path)
    data_files.remove("metadata.txt")


    if permute_data:
        perm_data_files = rng.permutation(data_files)
        train_files = perm_data_files[:train_size]
        test_files = perm_data_files[train_size:train_size+test_size]
    else:
        train_files = data_files[:train_size]
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
    Get real metadata as proper dict
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

def read_data(folder_path, file_name, col_names=None) -> pd.DataFrame:

    if col_names is None:
        col_names = ("time_ps","counts")

    file_path = os.path.join(folder_path, file_name)
    with open(file_path, "r", encoding="utf-8") as f:
        data = pd.read_table(f,sep=" ", names=col_names, skiprows=1, index_col=False)


    # print(file_name)
    # print(data)
    # print()

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
        data[i] = read_data(folder_path, file_name, col_names)["counts"].to_numpy()

    return data

def test_metadata():
    folder = os.path.join(os.getcwd(), "simdata")
    metadata = read_metadata(folder)
    print(metadata["simdata_00231.pals"])

def test_data():
    import matplotlib.pyplot as plt

    data = read_data("simdata","simdata_00999.pals")
    plt.plot(data["time_ps"], data["counts"])
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

def main():

    folder_path = os.path.join(os.getcwd(), "simdata")
    file_names = os.listdir(folder_path)
    # should only contain simdata files
    file_names.remove("metadata.txt")
    train = get_train_or_test(folder_path, file_names[:100])
    
    first = train[0]
    max_val_index = first.argmax()
    print(max_val_index)
    print(first[max_val_index:])

if __name__ == "__main__":

    test_read_real()