import marimo

__generated_with = "0.18.1"
app = marimo.App(
    width="full",
    layout_file="layouts/fm_radio_run_1_alveo_0_eigenvector_directions.slides.json",
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
def _(arrays, mo):
    ANTENNA_FLAGS = {
        0: [0,2,4,6],
        2: [],
        3:[0, 1, 5],
    }

    ANTENNA_POSITIONS = {
        0: [(-1, -1), (5.881,3.46), (-1, -1), (-16.773, 8.696), (-1, -1), (1.915,-18.699), (-1, -1), (14.896,12.985),  (2.064, -4.383), (3.587, -5.706),],
        1: [],
        2: [],
        3: [(0.482, 1.173), (2.203, 2.904), (5.851, 1.663), (4.064, 3.257), (2.152, 0.608), (3.71, 1.519), (-1.946, -3.131), (-1.028, -1.394), (-3.246, -1.023), (-1.925, 0.17)],
    }

    current_alveo = 3
    n_fine_channels = 16
    n_coarse_channels = 8

    BW_PER_COARSE_CHANNEL = 781.25 * 32 / 27
    BW_PER_FINE_CHANNEL = BW_PER_COARSE_CHANNEL / n_fine_channels


    GOOD_ANTENNAS = [_i for _i in range(10) if _i not in ANTENNA_FLAGS[current_alveo]]
    print(GOOD_ANTENNAS)
    ANTENNA_POS = ANTENNA_POSITIONS[current_alveo]
    ANTENNA_POS = [_a for _n, _a in enumerate(ANTENNA_POS) if _n in GOOD_ANTENNAS]
    array = arrays.Array(positions=ANTENNA_POS)
    print(array.positions)
    print(ANTENNA_POS)

    pcount_slider = mo.ui.slider(0, 200000, show_value=True,value=1000, step=1000, label="Number of packets to process:")
    pcount_slider

    pcount_slider = mo.ui.slider(0, 200000, show_value=True,value=1000, step=1000, label="Number of packets to process:")
    pcount_slider
    return (
        ANTENNA_POSITIONS,
        BW_PER_COARSE_CHANNEL,
        BW_PER_FINE_CHANNEL,
        GOOD_ANTENNAS,
        array,
        current_alveo,
        n_coarse_channels,
        n_fine_channels,
        pcount_slider,
    )


@app.cell
def _(ANTENNA_POSITIONS, GOOD_ANTENNAS, current_alveo):
    # generate all position swaps

    _pairs = [(ANTENNA_POSITIONS[current_alveo][_i], ANTENNA_POSITIONS[current_alveo][_i+1]) for _i in range(0, len(ANTENNA_POSITIONS[current_alveo]), 2)]
    _num_pairs = [(_i, _i+1) for _i in range(0, len(ANTENNA_POSITIONS[current_alveo]), 2)]
    import itertools

    def all_pair_swaps(pairs):
        choices = [
            (pair, pair[::-1])   # original or swapped
            for pair in pairs
        ]
        for combo in itertools.product(*choices):
            # flatten back to 10 points
            yield [pt for pair in combo for pt in pair]

    _all_pos_nums = list(all_pair_swaps(_num_pairs))

    _all_pos_lists = set()
    for _pos in _all_pos_nums:
        _output = []
        for _j in _pos:
            if _j in GOOD_ANTENNAS:
                _output.append(ANTENNA_POSITIONS[current_alveo][_j])
        _all_pos_lists.add(tuple(_output))

    len(_all_pos_lists)
    all_pos_lists = [list(_a) for _a in _all_pos_lists]
    return (all_pos_lists,)


@app.cell
def _(GOOD_ANTENNAS, file_browser, np, pcount_slider):
    import dpkt
    import struct
    import typing
    import argparse
    import math
    import matplotlib.pyplot as plt

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

    all_samples_scaled = all_samples_scaled[GOOD_ANTENNAS, : , :, :]
    print(all_samples_scaled.shape)
    return N_POL, all_samples_scaled, plt, seq_nums, start_chan, total_channels


@app.cell
def _(
    BW_PER_COARSE_CHANNEL,
    BW_PER_FINE_CHANNEL,
    all_samples_scaled,
    array,
    constants,
    n_fine_channels,
    np,
    plt,
    start_chan,
):
    import scipy
    def coarse_to_fine(samples, n_fine=n_fine_channels, n_discard_edge=1):
        """
        n_discard_edge : fine channels to flag at each coarse channel edge
                         due to 32/27 oversampling aliases (~15.6% each side)
        """
        n_ant, n_coarse, n_time, n_pol = samples.shape
        n_snapshots = n_time // n_fine

        # Periodic Hanning window
        #window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fine) / n_fine)

        samples = samples[:, :, :n_snapshots * n_fine, :]
        x = samples.reshape(n_ant, n_coarse, n_snapshots, n_fine, n_pol)

        Xf = np.fft.fft(x, axis=3)
        Xf = np.fft.fftshift(Xf, axes=3)  # centre DC

        fine = np.transpose(Xf, (0, 1, 3, 2, 4))

        # Flag/zero alias-contaminated edge channels
        if n_discard_edge > 0:
            fine[:, :, :n_discard_edge, :, :]  = 0
            fine[:, :, -n_discard_edge:, :, :] = 0

        return fine

    def calibrate_array(X, ref_ant=0, steering_vec = None, steering_vec_coarse_index=None, steering_vec_fine_index=None):

        n_ant, n_coarse, n_fine, n_time, n_pol = X.shape

        # Compute mean complex amplitude over time axis
        mean_amp = np.mean(X, axis=3)  # shape: n_ant, n_coarse, n_fine, n_pol
        # Compute per-antenna phase offsets relative to reference antenna
        phi_offsets = None
        if steering_vec is not None:
            print(mean_amp[:, steering_vec_coarse_index, steering_vec_fine_index, :].shape)
            actual_phases = np.angle(mean_amp[:, steering_vec_coarse_index, steering_vec_fine_index, :]) 
            print(f"actual phases: {actual_phases}")
            steering_phases = np.angle(steering_vec[:, None])
            phi_offsets = actual_phases - steering_phases
            print(f"phi_offsets are {phi_offsets}")
            print(f"steering_vector is {steering_vec}")
            print(f"steering vector phases are {steering_phases}")

           # phi_offsets = np.angle(mean_amp / steering_vec)
        else:
            phi_offsets = np.angle(mean_amp / mean_amp[ref_ant, :, :, :])
        #print(X.shape)
        # Remove phase offsets
        X_cal = X * np.exp(-1j * phi_offsets[:, None,None, None, :])

        # Normalize amplitudes per antenna
        #amp_gain = np.abs(mean_amp)

        #X_cal /= amp_gain[:, :, :, None, :] 

        return X_cal

    def get_phase_offsets(X, ref_ant=0, steering_vec = None, steering_vec_coarse_index=None, steering_vec_fine_index=None):

        n_ant, n_coarse, n_fine, n_time, n_pol = X.shape

        # Compute mean complex amplitude over time axis
        #mean_amp = np.mean(X, axis=3)  # shape: n_ant, n_coarse, n_fine, n_pol
        mean_amp = np.mean(X[:, steering_vec_coarse_index, steering_vec_fine_index, :, :] * np.conj(steering_vec[:, None, None]), axis=1)
        phase = np.angle(mean_amp)
        return phase

    def robust_calibrate_array(X_fine, start_chan, BW_PER_COARSE_CHANNEL, BW_PER_FINE_CHANNEL, ref_ant=0, steering_vec=None):
        """
        X_fine: shape (n_ant, n_coarse, n_fine, n_snapshots, n_pol)
        """
        n_ant, n_coarse, n_fine, n_snapshots, n_pol = X_fine.shape
    
        # 1. Compute mean complex amplitude over the snapshot (time) axis
        mean_amp = np.mean(X_fine, axis=3)  # shape: (n_ant, n_coarse, n_fine, n_pol)
    
        # 2. Flatten the frequency axes to 1D
        mean_amp_flat = mean_amp.reshape(n_ant, n_coarse * n_fine, n_pol)
        n_total_chans = n_coarse * n_fine
    
        # 3. Generate the flattened frequency axis (in Hz)
        freqs_hz = np.zeros(n_total_chans)
        for c in range(n_coarse):
            for f in range(n_fine):
                idx = c * n_fine + f
                # Convert your kHz formula to Hz
                freqs_hz[idx] = (781.25 * (start_chan + c) - (BW_PER_COARSE_CHANNEL)/2 + (BW_PER_FINE_CHANNEL)*0.5 + (BW_PER_FINE_CHANNEL)*f) * 1000
    
        # 5. Extract Delays via IFFT
        df = BW_PER_FINE_CHANNEL * 1000  # Channel spacing in Hz
        pad_factor = 8 # Zero-padding for sub-channel delay resolution
        n_padded = n_total_chans * pad_factor
        delay_axis = np.fft.fftfreq(n_padded, d=df)
    
        # Arrays to store our solved calibration terms
        taus = np.zeros((n_ant, n_pol))
        phase_dc = np.zeros((n_ant, n_pol))
    
        for pol in range(n_pol):
            # Cross-correlate with the reference antenna to get relative phase
            # vis shape: (n_ant, n_total_chans)
            ref_signal = mean_amp_flat[ref_ant, :, pol]
            vis = mean_amp_flat[:, :, pol] * np.conj(ref_signal[None, :])
        
            # Identify non-zero channels (ignoring the edges zeroed by coarse_to_fine)
            valid_mask = np.abs(ref_signal) > 0
        
            for ant in range(n_ant):
                if ant == ref_ant:
                    continue # tau and phase_dc remain 0
                
                # We only want to IFFT the valid channels. To keep the FFT uniform, 
                # the zeros from n_discard_edge actually act as a natural window, 
                # which is fine, but we pad it for resolution.
                padded_vis = np.pad(vis[ant, :], (0, n_padded - n_total_chans))
            
                # The Delay Transform
                delay_spectrum = np.fft.ifft(padded_vis)
                plt.plot(np.abs(delay_spectrum))
                # Find the peak delay
                peak_idx = np.argmax(np.abs(delay_spectrum))
                taus[ant, pol] = delay_axis[peak_idx]
            
                # Now find the DC phase offset. 
                # We know: Phase = -2 * pi * freqs * tau + phase_dc
                # So: phase_dc = Phase + 2 * pi * freqs * tau
                # We evaluate this over the valid channels and take the mean.
                actual_phases = np.angle(vis[ant, valid_mask])
                unwrapped_dc = actual_phases + 2 * np.pi * freqs_hz[valid_mask] * taus[ant, pol]
            
                # Wrap the mean DC phase back to [-pi, pi]
                phase_dc[ant, pol] = np.angle(np.mean(np.exp(1j * unwrapped_dc)))

        # 6. Reconstruct the ideal, noiseless calibration phase across all frequencies
        # shape: (n_ant, n_total_chans, n_pol)
        freqs_3d = freqs_hz[None, :, None]
        ideal_phase = -2 * np.pi * freqs_3d * taus[:, None, :] + phase_dc[:, None, :]
    
        # Reshape back to the original nested coarse/fine structure
        ideal_phase = ideal_phase.reshape(n_ant, n_coarse, n_fine, n_pol)
    
        # 7. Apply the calibration
        # X_fine shape: (n_ant, n_coarse, n_fine, n_snapshots, n_pol)
        # ideal_phase shape needs to broadcast over n_snapshots (axis 3)
        X_cal = X_fine * np.exp(-1j * ideal_phase[:, :, :, None, :])
    
        # Normalize amplitudes based on the reference antenna (optional, but good practice)
        amp_gain = np.abs(mean_amp[:, :, :, :])
        amp_gain = np.where(amp_gain == 0, 1, amp_gain) # Avoid divide-by-zero on flagged edges
        X_cal /= amp_gain[:, :, :, None, :] 
    
        return X_cal, taus, phase_dc


    def get_channel_center(coarse, fine):
        print(start_chan)
        _channel_center = (781.25 * (start_chan + coarse) - (BW_PER_COARSE_CHANNEL) / 2 + (BW_PER_FINE_CHANNEL) * 0.5  + (BW_PER_FINE_CHANNEL) * fine)
        return _channel_center



    all_samples_fine = coarse_to_fine(all_samples_scaled, n_fine=16)

    known_steering_vec_coarse_channel = 4
    known_steering_vec_fine_channel = 3
    known_channel_center = get_channel_center(known_steering_vec_coarse_channel, known_steering_vec_fine_channel) * 1000
    print(f"channel center: {known_channel_center}")
    print(f"channel bw: {BW_PER_FINE_CHANNEL} kHz")
    known_steering_vec_wv = (constants.c / known_channel_center)
    known_steering_vec_phi = 3.1 * np.pi / 180
    known_steering_vec_theta = 89 * np.pi / 180 
    known_steering_vec = array.steering_vector(
        [known_steering_vec_phi, known_steering_vec_theta], 
        known_steering_vec_wv
    )

    #all_samples_cal = calibrate_array(all_samples_fine,steering_vec=known_steering_vec, steering_vec_coarse_index=known_steering_vec_coarse_channel, steering_vec_fine_index=known_steering_vec_fine_channel)

    print(all_samples_fine.shape)




    return (
        all_samples_fine,
        get_channel_center,
        get_phase_offsets,
        robust_calibrate_array,
    )


@app.cell
def _(
    all_samples_fine,
    array,
    constants,
    get_channel_center,
    get_phase_offsets,
    np,
    plt,
):
    # Get phase offsets for first few coarse channels vs frequency
    # Altitude: 63.97°
    # Theta (Sph. Coord): 26.03°
    # Azimuth: 319.64°
    # Phi (Sph. Coord): 130.36°
    import pandas as pd
    _output = []
    freqs = np.zeros((8, 16))
    for _coarse_chan in range(8):
        for _fine_chan in range(2, 15):
            _chan_center = get_channel_center(_coarse_chan, _fine_chan) * 1000
            freqs[_coarse_chan, _fine_chan] = _chan_center


    for _coarse_chan in range(4,5):
        for _fine_chan in range(3, 14):
            _chan_center = get_channel_center(_coarse_chan, _fine_chan) * 1000
            _wv = (constants.c / _chan_center)
            _sun_phi =  130.3648 * np.pi / 180
            _sun_theta = 26.0294 * np.pi / 180 
            _steer_vec = np.conj(array.steering_vector(
                [_sun_phi, _sun_theta], 
                _wv
            ))
            _po = get_phase_offsets(all_samples_fine,steering_vec=_steer_vec, steering_vec_coarse_index=_coarse_chan, steering_vec_fine_index=_fine_chan)
            _out = {"freq": _chan_center}
            for _i in range(_po.shape[0]):
                _out['antenna'] = _i
                for _j in range(_po.shape[1]):
                    _out['pol'] = _j
                    _out['phase'] = _po[_i, _j]
                    _output.append(_out.copy())
                     
            #print(_po)

    _output = pd.DataFrame(_output)
    _output

    _fc = _output['freq'].mean()

    _output['freq_centered'] = _output['freq'] - _fc
    _output['phase_unwrapped'] = np.unwrap(_output['phase'])
    #_output[(_output['antenna'] == 5) & (_output['pol'] == 0)][['freq_centered', 'phase_unwrapped']].plot(x='freq_centered', y='phase_unwrapped')
    #a, b = np.polyfit(freq_centered, phase_unwrapped, 1)
    _subset = _output[(_output['antenna'] == 2) & (_output['pol'] == 1)]

    _n_ant = _output['antenna'].nunique()
    _n_pol = _output['pol'].nunique()

    a = np.zeros((_n_ant, _n_pol))
    b = np.zeros((_n_ant, _n_pol))

    for _ant in range(_n_ant):
        for _pol in range(_n_pol):

            _d = _output[
                (_output['antenna'] == _ant) &
                (_output['pol'] == _pol)
            ]

            _x = _d['freq_centered'].values
            _y = _d['phase_unwrapped'].values
            plt.plot(_x, _y)
            a[_ant,_pol], b[_ant,_pol] = np.polyfit(_x, _y, 1)


    fc = _fc
    _freq_centered = freqs - fc

    _phase = a[:,None,None,:] * _freq_centered[None,:,:,None] + b[:,None,None,:]

    corr = np.exp(-1j * _phase)

    all_samples_cal = all_samples_fine * corr[:,:,:,None,:]
    plt.gca()
    return all_samples_cal, pd


@app.cell
def _(
    BW_PER_COARSE_CHANNEL,
    BW_PER_FINE_CHANNEL,
    all_samples_fine,
    plt,
    robust_calibrate_array,
    start_chan,
):
    #all_samples_fine = coarse_to_fine(all_samples_scaled, n_fine=n_fine_channels)

    # 2. Calibrate using the robust delay method
    all_samples_cal_2, taus_2, dc_phases = robust_calibrate_array(
        all_samples_fine, 
        start_chan=start_chan,
        BW_PER_COARSE_CHANNEL=BW_PER_COARSE_CHANNEL, 
        BW_PER_FINE_CHANNEL=BW_PER_FINE_CHANNEL,
        ref_ant=3
    )

    print(f"Calibrated Array Shape: {all_samples_cal_2.shape}")
    print(f"Instrumental Delays (ns) for Pol 0:\n {taus_2[:, 0] * 1e9}")
    plt.gca()
    return (all_samples_cal_2,)


@app.cell
def _(all_samples_scaled, array, constants, np, pd, plt, start_chan):
    # Get phase offsets for first few coarse channels vs frequency
    # Altitude: 63.9706°
    # Theta (Sph. Coord): 26.0294°
    # Azimuth: 319.6352°
    # Phi (Sph. Coord): 130.3648°
    def get_phase_offsets_coarse(X, ref_ant=0, steering_vec = None, steering_vec_coarse_index=None):

        n_ant, n_coarse, n_time, n_pol = X.shape

        # Compute mean complex amplitude over time axis
        #mean_amp = np.mean(X, axis=2)  # shape: n_ant, n_coarse, n_fine, n_pol
        mean_amp = np.mean(X[:, steering_vec_coarse_index, :, :] * np.conj(steering_vec[:, None, None]), axis=1)
        phase = np.angle(mean_amp)
        # Compute per-antenna phase offsets relative to reference antenna
        #phi_offsets = None
    
        #actual_phases = np.angle(mean_amp[:, steering_vec_coarse_index, :]) 
        #steering_phases = np.angle(steering_vec[:, None])
        #phi_offsets = np.angle(np.exp(1j * (actual_phases - steering_phases)))
        #phi_offsets = actual_phases - steering_phases

        print(phase.shape)
        return phase



    _output = []
    coarse_freqs = np.zeros((8,))
    for _coarse_chan in range(8):
        _chan_center = 781.25 * (start_chan + _coarse_chan) * 1000
        coarse_freqs[_coarse_chan] = _chan_center


    for _coarse_chan in range(4,8):
        _chan_center = 781.25 * (start_chan + _coarse_chan) * 1000
        _wv = (constants.c / _chan_center)
        _sun_phi =  130.3648 * np.pi / 180
        _sun_theta = 26.0294 * np.pi / 180 
        _steer_vec = array.steering_vector(
            [_sun_phi, _sun_theta], 
            _wv
        )
        _po = get_phase_offsets_coarse(all_samples_scaled,steering_vec=_steer_vec, steering_vec_coarse_index=_coarse_chan)
        _out = {"freq": _chan_center}
        for _i in range(_po.shape[0]):
            _out['antenna'] = _i
            for _j in range(_po.shape[1]):
                _out['pol'] = _j
                _out['phase'] = _po[_i, _j]
                _output.append(_out.copy())
                     
            print(_po)

    _output = pd.DataFrame(_output)
    _output

    _fc = _output['freq'].mean()

    _output['freq_centered'] = _output['freq'] - _fc
    _output['phase_unwrapped'] = np.unwrap(_output['phase'])
    #_output[(_output['antenna'] == 5) & (_output['pol'] == 0)][['freq_centered', 'phase_unwrapped']].plot(x='freq_centered', y='phase_unwrapped')
    #a, b = np.polyfit(freq_centered, phase_unwrapped, 1)
    _subset = _output[(_output['antenna'] == 2) & (_output['pol'] == 1)]

    _n_ant = _output['antenna'].nunique()
    _n_pol = _output['pol'].nunique()

    _a = np.zeros((_n_ant, _n_pol))
    _b = np.zeros((_n_ant, _n_pol))

    for _ant in range(_n_ant):
        for _pol in range(_n_pol):

            _d = _output[
                (_output['antenna'] == _ant) &
                (_output['pol'] == _pol)
            ]

            _x = _d['freq_centered'].values
            _y = _d['phase_unwrapped'].values
            plt.plot(_x, _y)
            _a[_ant,_pol], _b[_ant,_pol] = np.polyfit(_x, _y, 1)

    _freq_centered = coarse_freqs - _fc

    _phase = _a[:,None,:] * _freq_centered[None,:,None] + _b[:,None,:]
    _corr = np.exp(-1j * _phase)

    all_samples_coarse_cal = all_samples_scaled * _corr[:,:,None,:]

    _tau = -_a / (2*np.pi)
    #plt.plot(_tau*1e9)
    #plt.ylabel("delay (ns)")
    #plt.xlabel("antenna")
    plt.gca()
    return


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
    GOOD_ANTENNAS,
    N_POL,
    all_samples_cal,
    all_samples_cal_2,
    np,
    total_channels,
    triangular_adc_pairs,
):
    _len_antennas = len(GOOD_ANTENNAS)
    _n_fine = all_samples_cal.shape[2]
    corr_mat = np.zeros((total_channels, _n_fine, int(_len_antennas * (_len_antennas + 1) / 2), N_POL, N_POL), dtype=np.complex64)
    pairs = triangular_adc_pairs(_len_antennas)

    for _c in range(total_channels):
        for _f in range(_n_fine):
            S = all_samples_cal_2[:, _c, _f, :, :]
            for (t_idx, (_i, _j)) in enumerate(pairs):
                corr_mat[_c, _f, t_idx] = S[_i].conj().T @ S[_j]  # shape: (ADCs, time, pol)  # S[i] = shape (time, pol)  # Outer-product over polarization:  #  #    Corr[p,q] = sum_t S[i,t,p] * conj(S[j,t,q])  #
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bandpass

    Shows the FFT for each channel. I believe the spikes are showing the pulse train injected into the stream by the FPGAS (approximately 300 KHz separation.)
    """)
    return


@app.cell
def _(
    BW_PER_FINE_CHANNEL,
    all_samples_cal,
    all_samples_fine,
    get_channel_center,
    np,
    plt,
    start_chan,
):
    # Fine channel bandpass
    print(all_samples_fine.shape)
    _nsamp = all_samples_fine.shape[3]
    _n_fine_chan = all_samples_fine.shape[2]
    (_fig, _axes) = plt.subplots(8, _n_fine_chan, figsize=(50, 12))
    _fig.subplots_adjust(top=0.8)

    _ant = 0

    # Frequency axis for fftshifted FFT (in kHz)
    _freqs = np.fft.fftshift(np.fft.fftfreq(_nsamp, d=1/BW_PER_FINE_CHANNEL))  # uses BW as effective sample rate
    _row_mags = []
    for _i in range(8):

        for _j in range(_n_fine_chan):
            _sig = all_samples_cal[_ant, _i, _j, :, 0]
            _fft_mag = np.abs(np.fft.fftshift(np.fft.fft(_sig)))
            _row_mags.append(_fft_mag)
        # Determine y-axis limits for this row
    _row_mags = np.array(_row_mags)
    _y_min = 0
    _y_max = np.nanmax(_row_mags)

    for _i in range(8):
        for _j in range(_n_fine_chan):
            _row = _i
            _col = _j
            _ax = _axes[_row, _col]

            # Extract signal and compute FFT magnitude
            _sig = all_samples_cal[_ant, _i, _j, :, 0]
            _fft_mag = np.abs(np.fft.fftshift(np.fft.fft(_sig)))

            # Plot magnitude
            _ax.plot(_freqs, _fft_mag)

            # Find peak
            _imax = np.argmax(_fft_mag)
            _fpeak = _freqs[_imax]
            _maxval = _fft_mag[_imax]

            _channel_center = get_channel_center(_i, _j)
            # Plot vline and annotate
            _ax.axvline(_fpeak, color='red', linestyle='--', linewidth=1)
            _ax.text(_fpeak, _maxval, f"{(_fpeak + _channel_center) / 1000:.2f} MHz", color='red',
                     rotation=0, va='bottom', ha='left', fontsize=8)
            _ax.set_ylim(_y_min, _y_max)

            # Title
            _ax.set_title(
                f"Chan {start_chan + _i}, Center = {_channel_center / 1000:.2f} MHz"
            )

            # Clean ticks
            _ax.set_xticks([])
       # _ax.set_yticks([])

    plt.tight_layout()
    plt.suptitle(f"Bandpass by Channel for Antenna {_ant}", y=1.01)

    plt.gca()
    return


@app.cell
def _(
    BW_PER_FINE_CHANNEL,
    GOOD_ANTENNAS,
    all_samples_cal,
    get_channel_center,
    np,
    plt,
    start_chan,
):
    # Bandpass of beamformed data

    _n_ant, _n_coarse, _n_fine, _nsamp, _n_pol = all_samples_cal.shape
    (_fig, _axes) = plt.subplots(_n_coarse, _n_fine, figsize=(24, 15))
    _fig.subplots_adjust(top=0.8)


    # Frequency axis for fftshifted FFT (in kHz)
    _freqs = np.fft.fftshift(np.fft.fftfreq(_nsamp, d=1/BW_PER_FINE_CHANNEL))  # uses BW as effective sample rate

    _row_mags = []
    for _i in range(8):
        for _j in range(_n_fine):
            _weights = np.ones((len(GOOD_ANTENNAS),1))
            _weights /= np.linalg.norm(_weights)
            _sig =_weights.T.conj() @ all_samples_cal[:, _i, _j, :, 0]
            _fft_mag = np.abs(np.fft.fftshift(np.fft.fft(_sig)))
            _row_mags.append(_fft_mag)
        # Determine y-axis limits for this row
    _row_mags = np.array(_row_mags)
    _y_min = 0
    _y_max = np.nanmax(_row_mags)

    for _i in range(8):
        for _j in range(_n_fine):
            _weights = np.ones((len(GOOD_ANTENNAS),1))
            _weights /= np.linalg.norm(_weights)
            _row = _i
            _col = _j
            _ax = _axes[_row, _col]

            # Extract signal and compute FFT magnitude
            _sig = _weights.T.conj() @ all_samples_cal[:, _i,_j, :, 0]

            # Extract signal and compute FFT magnitude
            _fft_mag = np.abs(np.fft.fftshift(np.fft.fft(_sig[0])))

            # Plot magnitude
            _ax.plot(_freqs, _fft_mag)

            # Find peak
            _imax = np.argmax(_fft_mag)
            _fpeak = _freqs[_imax]
            _maxval = _fft_mag[_imax]

            # Plot vline and annotate
            _ax.axvline(_fpeak, color='red', linestyle='--', linewidth=1)
            _ax.text(_fpeak, _maxval, f"{(_fpeak + get_channel_center(_i, _j)) / 1000:.2f} MHz", color='red',
                     rotation=0, va='bottom', ha='left', fontsize=8)

            # Title
            _ax.set_title(
                f"Chan {start_chan + _i}, Center = {get_channel_center(_i, _j) / 1000:.2f} MHz"
            )
            _ax.set_ylim(_y_min, _y_max)

            # Clean ticks
            _ax.set_xticks([])
            #_ax.set_yticks([])

    plt.tight_layout()
    plt.suptitle("Beamformed Bandpass by Channel", y=1.01)

    plt.gca()
    return


@app.cell
def _(
    BW_PER_FINE_CHANNEL,
    all_samples_cal,
    corr_mat_unpacked,
    get_channel_center,
    np,
    plt,
):
    _channel = 4
    _fine_channel = 6

    # Bandpass of beamformed data

    (_fig, _axes) = plt.subplots(3, 3, figsize=(12, 6))
    _fig.subplots_adjust(top=0.8)

    _nsamp = all_samples_cal.shape[3]

    # Frequency axis for fftshifted FFT (in kHz)
    _freqs = np.fft.fftshift(np.fft.fftfreq(_nsamp, d=1/BW_PER_FINE_CHANNEL))  # uses BW as effective sample rate

    _eval, _evecs = np.linalg.eigh(corr_mat_unpacked[_channel, _fine_channel, :, :, 0, 0])
    evecs = _evecs
    print(_evecs.shape)
    beams = {}

    for _eig in range(-corr_mat_unpacked.shape[2] , 0):
        _i = abs(_eig) - 1
        _weights = _evecs[:, _eig]
        print(np.linalg.norm(_weights))
        #for _j in ANTENNA_FLAGS[current_alveo]:
        #    _weights[_j] = 0

    # Extract signal and compute FFT magnitude
        _sig = _weights.T @ all_samples_cal[:, _channel, _fine_channel, :, 0]
        beams[_eig] = _sig
        _row = abs(_i) // 3
        _col = abs(_i) % 3
        _ax = _axes[_row, _col]

    # Extract signal and compute FFT magnitude
        _fft_mag = np.abs(np.fft.fftshift(np.fft.fft(_sig)))

    # Plot magnitude
        _ax.plot(_freqs, _fft_mag)

    # Find peak
        _imax = np.argmax(_fft_mag)
        _fpeak = _freqs[_imax]
        _maxval = _fft_mag[_imax]

    # Plot vline and annotate
        _ax.axvline(_fpeak, color='red', linestyle='--', linewidth=1)
        _ax.text(_fpeak, _maxval, f"{(_fpeak + get_channel_center(_channel, _fine_channel)) / 1000:.2f} MHz", color='red',
             rotation=0, va='bottom', ha='left', fontsize=8)

    # Title
        _ax.set_title(
        f"Beam Bandpass for Eigenvector {_i + 1}"
    )

    # Clean ticks
        _ax.set_xticks([])
    #_ax.set_yticks([])

    plt.tight_layout()
    plt.suptitle("Eigenvector-Beamformed Bandpass by Channel", y=1.01)

    plt.gca()
    return beams, evecs


@app.cell
def _(beams, np, plt):
    # Take lags
    _eig_1 = -1
    _eig_2 = -2
    ccf = np.correlate(beams[_eig_1], beams[_eig_2], mode='full') / len(beams[-1])
    ccf = ccf[len(beams[-1])-1:]

    ccf_mag = np.abs(ccf[0:200])

    # Find index of maximum
    max_idx = np.argmax(ccf_mag)
    max_val = ccf_mag[max_idx]

    print("Maximum magnitude:", max_val)
    print("Lag at maximum:", max_idx)

    plt.plot(np.abs(ccf[0:200]))
    plt.xlabel('Lag')
    plt.ylabel('Magnitude (arb)')
    plt.title(f"Lag for Eig {abs(_eig_1)} vs Eig {abs(_eig_2)}")
    plt.gca()
    #np.abs(ccf[0:20])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Direction Finding
    """)
    return


@app.cell
def _():
    # Lag 

    _sample_rate = 781.25 * 32/ 27 * 1000
    _time_between_samples = 1 / _sample_rate

    _c = 299_792_458 # m / s

    _distance_in_one_lag = _c * _time_between_samples
    _distance_in_one_lag # m
    return


@app.cell
def _():
    # _eig_vec = -1

    # def get_spatial_spectrum_for_channel(coarse_channel, fine_channel, evecs, ax):
    #     phi_scan = np.linspace(-np.pi, np.pi, 720)
    #     theta_scan = np.linspace(0, np.pi / 2, 360)

    #     wv = constants.c / (1000 *  get_channel_center(coarse_channel, fine_channel))

    #     P = np.zeros((len(phi_scan), len(theta_scan)))
    #     _ek = evecs[:, _eig_vec:]

    #     _noise_matrix = np.identity(len(GOOD_ANTENNAS)) - _ek.conj() @ _ek.T

    #     for _i, _phi_i in enumerate(phi_scan):
    #         for _j, _theta_i in enumerate(theta_scan):

    #             _a = array.steering_vector(
    #             [_phi_i, _theta_i],
    #             np.array([wv])
    #         ).flatten()
    #             P[_i, _j] = 1 / np.abs((_a.conj().T @ _noise_matrix @ _a))
    #         #P[_i, _j] = np.abs(_a.conj().T @ _ek)**2

    #     idx_max = np.unravel_index(np.argmax(P), P.shape)
    #     phi_hat = phi_scan[idx_max[0]]
    #     theta_hat = theta_scan[idx_max[1]]

    #     print(f"phi: {phi_hat / np.pi} pi = {phi_hat * 180 / np.pi} deg, theta: {theta_hat/np.pi} pi = {theta_hat * 180 / np.pi} degrees")

    #     _phi_deg = phi_scan * 180 / np.pi
    #     _theta_deg = theta_scan * 180 / np.pi

    #     _P_db = 10 * np.log10(P / np.max(P))
    #     ax.imshow(
    #     _P_db.T,
    #     extent=[
    #         _phi_deg.min(), _phi_deg.max(),
    #         _theta_deg.min(), _theta_deg.max()
    #     ],
    #     aspect="auto",
    #     origin="lower"
    #     )
    #     ax.set_xlabel("Azimuth φ (degrees)")
    #     ax.set_ylabel("Elevation θ (degrees)")


    # _fig, _axes = plt.subplots(n_coarse_channels, n_fine_channels, figsize=(24,12))

    # for _i in range(n_coarse_channels):
    #     for _j in range(n_fine_channels):
    #         if _j == 0 or _j == n_fine_channels - 1:
    #             continue
    #         _ax = _axes[_i, _j]
    #         _eval, _evecs = np.linalg.eigh(corr_mat_unpacked[_i, _j, :, :, 0, 0])
    #         get_spatial_spectrum_for_channel(_i, _j, _evecs, _ax)
    # #_ax.set_title("Spatial Spectrum (dB)")

    # #_fig.colorbar(_im, ax=_ax, label="Relative Power (dB)")

    # plt.gca()
    return


@app.cell
def _(
    array,
    constants,
    corr_mat_unpacked,
    get_channel_center,
    n_coarse_channels,
    n_fine_channels,
    np,
    plt,
):
    _eig_vec = -1

    def get_spatial_spectrum_for_channel_vec(coarse_channel, fine_channel, evecs, ax):
        phi_scan = np.linspace(-np.pi, np.pi, 4000)
        theta_scan = np.linspace(0, np.pi / 2, 720)
        wv = constants.c / (1000 * get_channel_center(coarse_channel, fine_channel))
        k = 2 * np.pi / wv

        _ek = evecs[:, :_eig_vec]
        _noise_matrix = _ek.conj() @ _ek.T

        # Build all unit vectors (3, N_phi, N_theta)
        PHI, THETA = np.meshgrid(phi_scan, theta_scan, indexing='ij')
        k_hat = np.array([
            np.cos(PHI) * np.sin(THETA),
            np.sin(PHI) * np.sin(THETA),
            np.cos(THETA)
        ])  # (3, N_phi, N_theta)

        # antenna_positions: (M, 3) — ENU coords of GOOD_ANTENNAS
        phase = k * (array.positions @ k_hat.reshape(3, -1))  # (M, N_grid)
        A = np.exp(-1j * phase)                                  # (M, N_grid)

        # MUSIC pseudospectrum — vectorized quadratic form
        tmp = _noise_matrix @ A                                          # (M, N_grid)
        denom = np.einsum('ij,ij->j', A.conj(), tmp)                    # (N_grid,)
        P = (1.0 / np.abs(denom)).reshape(len(phi_scan), len(theta_scan))

        idx_max = np.unravel_index(np.argmax(P), P.shape)
        phi_hat = phi_scan[idx_max[0]]
        theta_hat = theta_scan[idx_max[1]]
        print(f"phi: {phi_hat / np.pi:.3f} pi = {phi_hat * 180 / np.pi:.1f} deg, "
              f"theta: {theta_hat / np.pi:.3f} pi = {theta_hat * 180 / np.pi:.1f} deg")

        _phi_deg = phi_scan * 180 / np.pi
        _theta_deg = theta_scan * 180 / np.pi
        _P_db = 10 * np.log10(P / np.max(P))

        ax.imshow(
            _P_db.T,
            extent=[_phi_deg.min(), _phi_deg.max(), _theta_deg.min(), _theta_deg.max()],
            aspect="auto",
            origin="lower"
        )
        ax.set_xlabel("Azimuth φ (degrees)")
        ax.set_ylabel("Elevation θ (degrees)")


    _fig, _axes = plt.subplots(n_coarse_channels, n_fine_channels, figsize=(24, 12))
    for _i in range(n_coarse_channels):
        for _j in range(n_fine_channels):
            if _j == 0 or _j == n_fine_channels - 1:
                continue
            #print(f"Chan {_i}, {_j}")
            _ax = _axes[_i, _j]
            _eval, _evecs = np.linalg.eigh(corr_mat_unpacked[_i, _j, :, :, 0, 0])
            get_spatial_spectrum_for_channel_vec(_i, _j, _evecs, _ax)

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(GOOD_ANTENNAS, all_pos_lists, arrays, constants, evecs, np, plt):
    _fig, _axes = plt.subplots(len(all_pos_lists) // 2, 2)

    def generate_music_plot(positions, counter_val):
        array = arrays.Array(positions=positions)
        print(positions)
        _channel = 4

        # phi_scan = np.linspace(0, np.pi/2, 50)
        # theta_scan = np.linspace(-1 * np.pi, np.pi, 100)
        wv = constants.c / 91300000.0#(117 * 781.25 * 1000)

        print(constants.c / wv)
        # results = []
        # for phi_i in phi_scan:
        #     for theta_i in theta_scan:
        #         w = array.steering_vector([phi_i, theta_i], np.array([constants.c / mid_freq]))
        #         X_weighted = w.conj().T @ all_samples_scaled[:, _channel, :, 0]
        #         results.append(10 * np.log10(np.var(X_weighted)))
        # results -= np.max(results)
        phi_scan = np.linspace(-np.pi, np.pi, 400)
        theta_scan = np.linspace(0, np.pi / 2, 400)

        P = np.zeros((len(phi_scan), len(theta_scan)))
        _ek = evecs[:, -1]
        print(_ek)
        print(_ek.shape)
        print(f"_ek norm is {np.linalg.norm(_ek)}")

        _noise_matrix = np.identity(len(GOOD_ANTENNAS)) - _ek.conj() @ _ek.T

        for _i, _phi_i in enumerate(phi_scan):
            for _j, _theta_i in enumerate(theta_scan):

                _a = array.steering_vector(
                    [_phi_i, _theta_i],
                    np.array([wv])
                ).flatten()
                P[_i, _j] = 1 / np.abs((_a.conj().T @ _noise_matrix @ _a))
                #P[_i, _j] = np.abs(_a.conj().T @ _ek)**2

        idx_max = np.unravel_index(np.argmax(P), P.shape)
        phi_hat = phi_scan[idx_max[0]]
        theta_hat = theta_scan[idx_max[1]]

        # _fig, _ax = plt.subplots(1, 1)
        # _ax.plot(theta_scan * 180 / np.pi, results)  # lets plot angle in degrees
        # _ax.set_xlabel("Theta [Degrees]")
        # _ax.set_ylabel("DOA Metric")
        # _ax.grid()

        print(f"phi: {phi_hat / np.pi} pi = {phi_hat * 180 / np.pi} deg, theta: {theta_hat/np.pi} pi = {theta_hat * 180 / np.pi} degrees")

        _phi_deg = phi_scan * 180 / np.pi
        _theta_deg = theta_scan * 180 / np.pi

        _P_db = 10 * np.log10(P / np.max(P))

        _ax = _axes[counter_val // 2, counter_val % 2]

        _im = _ax.imshow(
            _P_db.T,
            extent=[
                _phi_deg.min(), _phi_deg.max(),
                _theta_deg.min(), _theta_deg.max()
            ],
            aspect="auto",
            origin="lower"
        )

        _ax.set_xlabel("Azimuth φ (degrees)")
        _ax.set_ylabel("Elevation θ (degrees)")
        #_ax.set_title("Spatial Spectrum (dB)")


    # for _counter, _pos in enumerate(all_pos_lists):
    #     generate_music_plot(_pos, _counter)
    # #_fig.colorbar(_im, ax=_ax, label="Relative Power (dB)")
    # plt.gca()
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
def _(corr_mat, np):

    def unpack_triangular_corr(data):
        """
        data shape: (T, C, B, P1, P2, 2)
        where B = N(N+1)/2
        returns: (T, C, F, N, N, P1, P2)
        """
        (C, F, B, P1, P2) = data.shape
        N = int((np.sqrt(8 * B + 1) - 1) // 2)
        assert N * (N + 1) // 2 == B  # infer N
        k = np.arange(B)
        Tn = np.arange(N + 1)
        Tn = Tn * (Tn + 1) // 2  # convert real/imag → complex
        _a = np.searchsorted(Tn, k + 1) - 1
        start = Tn[_a]
        _b = k - start  # correct (a, b) mapping with no overflow
        R = np.zeros((C,F, N, N, P1, P2), dtype=data.dtype)
        R[:, :, _a, _b, :, :] = data
        R[:, :, _b, _a, :, :] = np.conj(data)  # triangular sequence
        return R  # ensures a < N+1 always  # output: full Hermitian matrix  # fill lower triangle  # fill upper using Hermitian symmetry



    corr_mat_unpacked = unpack_triangular_corr(corr_mat)
    print(corr_mat_unpacked.shape)
    return (corr_mat_unpacked,)


@app.cell
def _(all_samples_scaled):

    print(all_samples_scaled.shape)
    return


@app.cell
def _():
    import numpy as np
    from extraterrena import arrays, constants
    return arrays, constants, np


@app.cell
def _():
    return


@app.cell
def _():
    from astropy.coordinates import EarthLocation, AltAz, get_sun
    from astropy.time import Time
    import astropy.units as u

    def sun_position(lat, lon, elevation_m, datetime_utc):
        """
        Compute solar azimuth and altitude.

        lat, lon: degrees
        elevation_m: meters
        datetime_utc: ISO string or datetime (UTC)
        """

        location = EarthLocation(
            lat=lat * u.deg,
            lon=lon * u.deg,
            height=elevation_m * u.m
        )

        time = Time(datetime_utc)

        altaz_frame = AltAz(obstime=time, location=location)

        sun = get_sun(time).transform_to(altaz_frame)

        return {
            "altitude_deg": sun.alt.degree,
            "azimuth_deg": sun.az.degree
        }

 

    lat = -30.307665436   # Paul Wild
    lon = 149.550164466
    elevation = 212   # meters
    # local time is 14:22 on 2026-02-24
    _time = "2026-02-24T03:22:00"  # UTC

    pos = sun_position(lat, lon, elevation, _time)

    print(f"Altitude: {pos['altitude_deg']:.4f}°")
    print(f"Theta (Sph. Coord): {90 -pos['altitude_deg']:.4f}°")
    print(f"Azimuth: {pos['azimuth_deg']:.4f}°")
    print(f"Phi (Sph. Coord): {90 + 360 - pos['azimuth_deg']:.4f}°")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
