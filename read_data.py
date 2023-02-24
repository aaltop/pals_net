import json
import pandas as pd
import os
import numpy as np

def read_metadata(folder_path, metadata_name=None) -> dict:

    if metadata_name is None:
        metadata_name = "metadata.txt"

    metadata_path = os.path.join(folder_path, metadata_name)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    
    return metadata

def get_components(metadata:dict, file_names:list) -> list:
    '''
    Get lifetime and intensity components for given
    data determined by <file_names>, a list of data filename strings. 
    <metadata> can be fetched with the use of the function read_metadata.

    Returns components in a list as flattened numpy arrays.
    '''

    all_comps = [0]*len(file_names)

    for i,file_name in enumerate(file_names):
        all_comps[i] = np.array(
            metadata[file_name]["components"]).flatten()

    return all_comps

def read_data(folder_path, file_name) -> pd.DataFrame:

    file_path = os.path.join(folder_path, file_name)
    with open(file_path, "r", encoding="utf-8") as f:
        data = pd.read_table(f,sep=" ")

    return data

def get_train_or_test(folder_path:str, file_names):
    '''
    Returns the counts for the data in the files <file_names>
    from the folder <folder_path>.

    '''
    
    data = [0]*len(file_names)

    for i,file_name in enumerate(file_names):
        data[i] = read_data(folder_path, file_name)["counts"].to_numpy()

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

def main():

    folder_path = os.path.join(os.getcwd(), "simdata")
    file_names = os.listdir(folder_path)
    # should only contain simdata files
    file_names.remove("metadata.txt")
    train = get_train_or_test(folder_path, file_names[:100])
    print(len(train), len(train[0]))

if __name__ == "__main__":

    main()