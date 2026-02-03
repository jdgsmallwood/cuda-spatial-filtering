import marimo

__generated_with = "0.18.1"
app = marimo.App(
    width="full",
    layout_file="layouts/correlate-pcap.slides.json",
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # PCAP Correlation / Time Series View

    ## How to use
    1. Select the PCAP file you wish to view in the file browser below.
    2. Select how many packets you want to unpack from the file (NB: For Lambda, 115000 pkts is 1 second of data. I recommend ~1000 pkts for general viewing)
    3. All graphs / tables will update automatically.
    """)
    return


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    file_browser = mo.ui.file_browser(
        initial_path=Path("/home/jay/projects/cuda-spatial-filtering/scripts"), multiple=False
    )

    # Access the selected file path(s):
    file_browser
    return file_browser, mo


@app.cell
def _(mo):
    pcount_slider = mo.ui.slider(0, 200000, show_value=True,value=1000, step=1000, label="Number of packets to process:")
    pcount_slider
    return (pcount_slider,)


@app.cell
def _(file_browser, pcount_slider):
    import dpkt
    import struct
    import typing
    import argparse
    import math
    import matplotlib.pyplot as plt
    import numpy as np
    import sys
    import time
    from collections import defaultdict
    from tqdm import tqdm


    def get_udp_payload_bytes(pcap_filename, packets = 900) -> typing.List[bytes]:
        """
        Read UDP payload bytes from a .pcap(ng) file
        return list with one entry per packet, entry containing udp payload
        """
        timestamps = []
        try:
            with open(pcap_filename, "rb") as f:
                pcap_read = dpkt.pcap.UniversalReader(f)
                udp_payloads = list()
                pkt_count = 0
                for ts, buf in pcap_read:
                    # skip packet if not long enough to contain IP+UDP+CODIF hdrs
                    if len(buf) < (34 + 8 + 64):
                        print(f"WARNING: Found packet that is too small {len(buf)}Bytes")
                        continue
                    eth = dpkt.ethernet.Ethernet(buf)
                    ip = eth.data
                    # skip non-UDP packets
                    if ip.p != dpkt.ip.IP_PROTO_UDP:
                        print(f"WARNING: Found packet that is not UDP {ip.p} type")
                        continue
                    # add the UDP payload data into the list of payloads
                    udp = ip.data
                    udp_payloads.append(udp.data)
                    timestamps.append(ts)
                    pkt_count+=1
                    if pkt_count == packets:
                        break
        except FileNotFoundError as fnf_err:
            print(fnf_err)
            sys.exit(1)

        return timestamps, udp_payloads


    N_POL = 2
    N_VALS_PER_CPLX = 2
    N_BYES_PER_VAL = 1
    N_BYTES_PER_SAMPLE = N_POL * N_VALS_PER_CPLX * N_BYES_PER_VAL


    def parse_args():
        parser = argparse.ArgumentParser(prog="pss packet capture analyser")
        parser.add_argument("-f", "--file", required=True, help="File to analyse")
        parser.add_argument("-t", "--txt", type=str, help="Title text")
        return parser.parse_args()



    #arg_parser = parse_args()

    lambda_file = file_browser.path(index=0)
    total_ADCs = 10
    pcount_max = pcount_slider.value

    # read pcap file
    tstamps, payloads = get_udp_payload_bytes(lambda_file, pcount_max)

    # run through the data to find the number of beams and channels
    first_pkt = True
    start_seq_no = 0
    start_chan = 0
    end_seq_no = 0
    end_chan = 0
    total_packets = 0

    for ts, pkt_payload in zip(tstamps, payloads):
        seq_no = struct.unpack("<Q", pkt_payload[0:8])[0]
        FPGA_id = struct.unpack("<I", pkt_payload[8:12])[0]
        freq_chan = struct.unpack("<H", pkt_payload[12:14])[0]
        total_packets += 1
        if first_pkt:
            first_pkt = False
            start_seq_no = seq_no
            end_seq_no = seq_no
            start_chan = freq_chan
            end_chan = freq_chan
        else:
            if freq_chan < start_chan:
                start_chan = freq_chan
            if seq_no < start_seq_no:
                start_seq_no = seq_no
            if freq_chan > end_chan:
                end_chan = freq_chan
            if seq_no > end_seq_no:
                end_seq_no = seq_no
        if total_packets == pcount_max:
            break

    print(f"Found {total_packets} packets")
    print(
        f"Start time sample = {start_seq_no}, total time samples = {end_seq_no - start_seq_no + 1}, (= total time {1080e-9 * (end_seq_no - start_seq_no + 1)} seconds)"
    )
    print(
        f"Start channel = {start_chan}, total channels = {(end_chan - start_chan) + 1}"
    )
    total_channels = (end_chan - start_chan) + 1
    total_time_packets = (end_seq_no - start_seq_no + 1) // 64
    expected_packets = total_channels * total_time_packets
    print(f"expected packets = {expected_packets}")

    # Get all the data into a big numpy array
    # ADCs x channels x time samples
    all_samples = np.zeros(
        (total_ADCs, total_channels, (end_seq_no - start_seq_no + 64), N_POL),
        dtype=np.complex64,
    )
    all_samples_scaled = np.zeros(
        (total_ADCs, total_channels, (end_seq_no - start_seq_no + 64), N_POL),
        dtype=np.complex64,
    )
    # scale factor for each sample, initialise with -1
    all_scales = -500 * np.ones(
        (total_ADCs, total_channels, (end_seq_no - start_seq_no + 64), N_POL), dtype=np.float32
    )
    pkt_scale = np.zeros((total_ADCs, N_POL), dtype=np.float32)
    pcount = 0
    seq_nums = defaultdict(list)
    for ts, pkt_payload in tqdm(zip(tstamps, payloads), total=expected_packets):
        pcount += 1
        seq_no = struct.unpack("<Q", pkt_payload[0:8])[0]
        FPGA_id = struct.unpack("<I", pkt_payload[8:12])[0]
        freq_chan = struct.unpack("<H", pkt_payload[12:14])[0] - start_chan
        seq_nums[freq_chan].append(seq_no)
        padding = struct.unpack("<Q", pkt_payload[14:22])[0]

        # Get scale factors
        for adc in range(total_ADCs):
            for p in range(N_POL):
                pkt_scale[adc][p] = np.float32(
                struct.unpack("H", pkt_payload[(22 + 2 * adc * N_POL + 2 * p) : (24 + 2 * adc * N_POL + 2 * p)])[0]
            )
        # Get data
        data_base = 22 + total_ADCs * 2 * N_POL
        for adc in range(total_ADCs):
            for p in range(N_POL):
                for t in range(64):  # 64 time samples per packet
                    x_i, x_q = struct.unpack(
                        "bb",
                        pkt_payload[
                            (data_base + t * total_ADCs * N_POL * 2 + adc * 2 * N_POL + 2 * p) : (
                                data_base + t * total_ADCs *N_POL * 2 + adc * 2 * N_POL + 2 * p + 2
                            )
                        ],
                    )
                    all_samples[adc, freq_chan, seq_no - start_seq_no + t, p] = (
                        1j * np.float32(x_q) + np.float32(x_i)
                    )
                    all_samples_scaled[adc, freq_chan, seq_no - start_seq_no + t,p] = (
                        pkt_scale[adc][p] * (1j * np.float32(x_q) + np.float32(x_i))
                    )
                    all_scales[adc, freq_chan, seq_no - start_seq_no + t,p] = pkt_scale[adc][p]
        if pcount >= pcount_max:
           print(f"stopping packet decoding at packet {pcount}")
           break
    return (
        N_POL,
        all_samples,
        all_samples_scaled,
        np,
        plt,
        seq_nums,
        total_ADCs,
        total_channels,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checking for missing packets
    """)
    return


@app.cell
def _(seq_nums):
    for (chan, seqs) in seq_nums.items():
        for (_i, seq) in enumerate(seqs):
            if _i == 0:
                continue
            if seq - seqs[_i - 1] != 64:
                print(f'Diff at chan {chan} packet {_i} has sequence number {seq} when expected was {seqs[_i - 1] + 64}. Number of missing packets is {(seq - seqs[_i - 1] - 64) / 64}')
    return


@app.cell
def _(
    N_POL,
    all_samples_scaled,
    np,
    total_ADCs,
    total_channels,
    triangular_adc_pairs,
):
    corr_mat = np.zeros((total_channels, int(total_ADCs * (total_ADCs + 1) / 2), N_POL, N_POL), dtype=np.complex64)
    pairs = triangular_adc_pairs(total_ADCs)
    for f in range(total_channels):
        S = all_samples_scaled[:, f, :, :]
        for (t_idx, (_i, j)) in enumerate(pairs):
            corr_mat[f, t_idx] = S[_i].conj().T @ S[j]  # shape: (ADCs, time, pol)  # S[i] = shape (time, pol)  # Outer-product over polarization:  #  #    Corr[p,q] = sum_t S[i,t,p] * conj(S[j,t,q])  #
    return (corr_mat,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Correlation Matrix Shape

    (channels, baseline, pol, pol)

    baselines are triangular
    0: 0-0
    1: 0-1
    2: 1-1
    3: 0-2
    4: 1-2
    5: 2-2
    etc.
    """)
    return


@app.cell
def _(corr_mat):
    corr_mat.shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Preview of correlation matrix for channel 0, first six baselines, pol 0
    """)
    return


@app.cell
def _(corr_mat):
    corr_mat[0, :6, 0,0]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Time Samples Shape
    (receivers, channels, samples, polarizations)

    Use the sliders underneath the graph to control channel, receiver, and polarization.
    """)
    return


@app.cell
def _(all_samples_scaled):
    all_samples_scaled.shape
    return


@app.cell
def _(
    all_samples_scaled,
    plt,
    time_series_channel_slider,
    time_series_pol_slider,
    time_series_receiver_slider,
):
    _channel = time_series_channel_slider.value
    _receiver = time_series_receiver_slider.value
    _pol = time_series_pol_slider.value
    plt.plot(all_samples_scaled[_receiver,_channel,:,_pol].real)
    plt.plot(all_samples_scaled[_receiver,_channel,:,_pol].imag)
    plt.title(f"Time series of channel {_channel} and receiver {_receiver} and pol {_pol}")
    plt.gca()
    return


@app.cell
def _(mo):
    time_series_channel_slider = mo.ui.slider(start=0, stop=7, step=1, show_value=True, label="Channel to show:")
    time_series_receiver_slider = mo.ui.slider(start=0,stop=9, step=1, show_value=True, label="Receiver to show:")
    time_series_pol_slider = mo.ui.slider(start=0, stop=1, step=1, show_value=True, label="Pol to show:")
    mo.vstack([time_series_channel_slider, time_series_receiver_slider, time_series_pol_slider])
    return (
        time_series_channel_slider,
        time_series_pol_slider,
        time_series_receiver_slider,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bandpass

    Shows the FFT for each channel. I believe the spikes are showing the pulse train injected into the stream by the FPGAS (approximately 300 KHz separation.)
    """)
    return


@app.cell
def _(all_samples_scaled, np, plt):
    (fig, axes) = plt.subplots(2, 4, figsize=(10, 6))
    for _i in range(8):
        row = _i // 4
        col = _i % 4
        axes[row, col].plot(np.fft.fftshift(np.abs(np.fft.fft(all_samples_scaled[0, _i, :, 0]))))
        axes[row, col].set_title(f'i = {_i}')
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
    plt.tight_layout()
    plt.suptitle("Bandpass by Channel")

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Diagnostics

    ### Histograms of sample values
    Sample values are signed int8_t which has a range of -128 to 127.
    Should be Gaussian w/ mean 0.
    """)
    return


@app.cell
def _(all_samples, plt):
    plt.hist((all_samples[:,:,:,:].real).flatten(), bins=100)
    plt.title("Histogram of sample value (real, before scaling)")
    plt.gca()
    return


@app.cell
def _(all_samples, plt):
    plt.hist((all_samples[:,:,:,:].imag).flatten(), bins=100)
    plt.title("Histogram of sample value (imag, before scaling)")
    plt.gca()
    return


@app.cell
def _(np):
    def triangular_adc_pairs(N):
        """Return list of (i, j) index pairs for lower-triangle storage."""
        pairs = []
        for _i in range(N):
            for j in range(_i + 1):
                pairs.append((_i, j))
        return pairs

    def hermitian_from_lower_triangular(vec, n):
        """
        vec: 1D array of length n(n+1)/2 containing the LOWER triangle row-by-row.
        n: size of the Hermitian matrix.
        """
        H = np.zeros((n, n), dtype=complex)
        idx = 0
        for _i in range(n):
            for j in range(_i + 1):
                H[_i, j] = vec[idx]
                idx += 1
        H = H + np.tril(H, -1).conj().T
        return H
    return (triangular_adc_pairs,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
