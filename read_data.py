import json
import pandas as pd
import os

def read_metadata(folder_path, metadata_name=None):

    if metadata_name is None:
        metadata_name = "metadata.txt"

    metadata_path = os.path.join(folder_path, metadata_name)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    
    return metadata

def read_data(folder_path, file_name):

    file_path = os.path.join(folder_path, file_name)
    with open(file_path, "r", encoding="utf-8") as f:
        data = pd.read_table(f,sep=" ")

    return data

def test_metadata():
    folder = os.path.join(os.getcwd(), "simdata")
    metadata = read_metadata(folder)
    print(metadata["simdata_00231.pals"])

def main():
    import matplotlib.pyplot as plt

    data = read_data("simdata","simdata_00999.pals")
    plt.plot(data["time_ps"], data["counts"])
    plt.yscale("log")
    plt.show()



if __name__ == "__main__":

    main()