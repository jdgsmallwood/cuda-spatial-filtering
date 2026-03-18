import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    from pathlib import Path
    file_browser = mo.ui.file_browser(
        initial_path=Path("/Users/jsmallwood/projects/cuda-spatial-filtering/scripts"), multiple=False
    )

    # Access the selected file path(s):
    file_browser
    return (file_browser,)


@app.cell
def _(file_browser):
    import h5py
    import numpy as np
    #file_browser.
    file_paths = file_browser.value 
    num_data_sets = len(file_paths)

    data_arrays = []
    for path in file_paths:
        with h5py.File(path.path, "r") as hdf:
            print(f"Opening {path.path}")

            # List all groups in the file
            print("Keys in the file:")
            for key in hdf.keys():
                print(" -", key)
            dataset_name = "projection_eigenvalues"
            if dataset_name in hdf:
                print(hdf[dataset_name][:].shape)
                data_arrays.append(hdf[dataset_name][:60])
                print(f"\nData from '{dataset_name}':")
                #print(data)
            else:
                print(f"\nDataset '{dataset_name}' not found in the file.")
            if "vis_missing_nums" in hdf:
                print("Vis missing nums:")

                print(hdf["vis_missing_nums"][:])
    data = np.concat(data_arrays)
    return (data,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(data):
    data.shape
    return


@app.cell
def _(data):
    import matplotlib.pyplot as plt

    _data = data[0][4][0][0]

    _top_3_eigs = data[:, 4, 0,0,-3:]
    #plt.plot(_data)
    plt.gca()

    print(_data)
    plt.plot(_top_3_eigs)
    plt.gca()

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
