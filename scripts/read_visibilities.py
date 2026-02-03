import marimo

__generated_with = "0.18.1"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # LAMBDA Visibilities

    ## How to use
    1. Select the hdf5 files you wish to open (you can select multiple using Shift, but ensure you select them in time order)
    2. Graphs and tables will update automatically.

    ## Joining files
    When multiple files are selected, they will be concatenated across the time index. Each file contains ~60 seconds of data in 60 x 1sec integrations.
    """)
    return


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    file_browser = mo.ui.file_browser(
        initial_path=Path("/home/jay/projects/cuda-spatial-filtering/build/apps"), multiple=True
    )

    # Access the selected file path(s):
    file_browser
    return file_browser, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data Shape

    Initial shape:
    (time x channels x baseline x pol x pol x real/imag)

    Baselines are triangular
    0: 0-0
    1: 0-1
    2: 1-1
    3: 0-2
    4: 1-2
    5: 2-2
    etc.

    This gets reshaped to
    (time x channels x receiver x receiver x pol x pol) complex entries.

    Initial baselines are 32 * (33) / 2 = 528 as we pad up to 32 receivers for the GPU pipeline.

    All baselines after 10 * 11 / 2 = 55 are zero so can be stripped out.
    """)
    return


@app.cell
def _(data):
    data.shape
    return


@app.cell
def _(d):
    d.shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Below is a pandas dataframe version of the correlation matrix and can be filtered and sorted by hovering over the column headers.
    """)
    return


@app.cell
def _(d, np):
    import pandas as pd

    # generate index columns
    coords = np.indices(d.shape).reshape(d.ndim, -1).T

    long_df = pd.DataFrame(coords, columns=["time_segment", "channel", "receiver_0", "receiver_1", "pol", "pol_2"])
    long_df["vis"] = d.ravel()
    long_df["mag"] = long_df["vis"].abs()
    long_df["angle"] = np.angle(long_df['vis'])
    long_df['autocorr'] = (long_df['receiver_0'] == long_df['receiver_1']) & (long_df['pol'] == long_df['pol_2']) 
    long_df
    return


@app.cell
def _(data, unpack_triangular_corr):
    _pol = 0
    _channel = 4
    _t_samp = 0
    d = unpack_triangular_corr(data)
    d = d[:, :, 0:10, 0:10, :, :]
    d[_t_samp, _channel, 0:10, 0:10, _pol, _pol]
    return (d,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plots

    For the first time sample, plots magnitude vs channel for all baselines.
    Legend not included as it is huge and unhelpful.
    """)
    return


@app.cell
def _(d, np):

    import matplotlib.pyplot as plt
    _mag = np.abs(d)
    _x = np.arange(8)
    fig = plt.figure(figsize=(10, 8))
    for _idx in range(0, 10):
        for _a in range(0, 10):
            if _idx != _a and _a < _idx:
                continue
            for _b in range(2):
                for _c in range(2):
                    if _b <= _c:
                        _y = np.log10(_mag[0, :, _idx, _a, _b, _c])
                        if np.any(_y != 0):
                            plt.plot(_x, _y, label=f'R{_idx}-R{_a}-P{_b}-P{_c}')

    plt.xlabel('Channel')
    plt.ylabel('Log10 Magnitude')
    plt.title('Magnitude')
    plt.tight_layout()
    plt.gca()
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Phase vs channel for the first time sample.
    """)
    return


@app.cell
def _(d, np, plt):
    _x = np.arange(8)
    plt.figure(figsize=(12, 8))
    for _idx in range(0, 10):
        for _a in range(0, 10):
            if _idx != _a and _a < _idx:
                continue
            for _b in [0,1]:
                for _c in [0,1]:
                    if _b <= _c:
                        _y = np.angle(d[0, :, _idx, _a, _b, _c])
                        if np.any(_y != 0):
                            plt.plot(_x, _y, label=f'R{_idx}-R{_a}-P{_b}-P{_c}')

    plt.xlabel('Channel')
    plt.ylabel('Phase')
    plt.title('Phase')
    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Amplitude averaged over all channels for first baseline:
    """)
    return


@app.cell
def _(d, np, plt):
    amp = np.abs(d[:, :, 0,1, 0, 0])  # baseline 0, pol 0

    plt.plot(amp.mean(axis=1))  # average over channels
    plt.xlabel("Time")
    plt.ylabel("Visibility amplitude")
    plt.title("Baseline amplitude vs time")
    plt.gca()
    return


@app.cell
def _(phase, plt, slider):

    plt.figure(figsize=(20,10))
    plt.plot(phase[slider.value, :])  
    plt.title("Phase vs time")
    plt.gca()
    return


@app.cell
def _(d, mo, np):
    phase = np.angle(d[:, :, 0, 9, 0, 0])  
    slider = mo.ui.slider(start=0, stop=phase.shape[0] -1, step=1, label="Time:", show_value=True)

    slider
    return phase, slider


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Shows the eigenvalues and eigenvectors of the correlation matrix.
    """)
    return


@app.cell
def _(d, np):
    _pol = 0
    _channel = 1
    _t_samp = 0
    (evals, evecs) = np.linalg.eigh(d[_t_samp, _channel, :10, :10, _pol, _pol])
    return evals, evecs


@app.cell
def _(evals):
    evals
    return


@app.cell
def _(evecs):
    # Most powerful eigenvector
    evecs[:,-1]
    return


@app.cell
def _(d, np):
    _t_samp = 0
    _channel = 0
    _pol = 0

    channel_evecs = []
    for _channel in range(8):
        (evals_1, evecs_1) = np.linalg.eigh(d[_t_samp, _channel, :, :, _pol, _pol])
        dom = evecs_1[:, -1]
        channel_evecs.append(dom)
    return


@app.cell
def _(np):

    def unpack_triangular_corr(data):
        """
        data shape: (T, C, B, P1, P2, 2)
        where B = N(N+1)/2
        returns: (T, C, N, N, P1, P2)
        """
        (T, C, B, P1, P2, _) = data.shape
        N = int((np.sqrt(8 * B + 1) - 1) // 2)
        assert N * (N + 1) // 2 == B  # infer N
        d = data[..., 0] + 1j * data[..., 1]
        k = np.arange(B)
        Tn = np.arange(N + 1)
        Tn = Tn * (Tn + 1) // 2  # convert real/imag → complex
        _a = np.searchsorted(Tn, k + 1) - 1
        start = Tn[_a]
        _b = k - start  # correct (a, b) mapping with no overflow
        R = np.zeros((T, C, N, N, P1, P2), dtype=d.dtype)
        R[:, :, _a, _b, :, :] = d
        R[:, :, _b, _a, :, :] = np.conj(d)  # triangular sequence
        return R  # ensures a < N+1 always  # output: full Hermitian matrix  # fill lower triangle  # fill upper using Hermitian symmetry
    return (unpack_triangular_corr,)


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
            dataset_name = "visibilities"
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
    return data, np


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
