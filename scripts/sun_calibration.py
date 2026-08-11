import marimo

__generated_with = "0.18.1"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # LAMBDA Sun Calibration

    Loads a visibility HDF5 file and compares observations against a geometric
    point-source model for the Sun. For each baseline the expected visibility
    is `V_model = exp(2πi f/c · (l Δe + m Δn + n Δu))` where (l, m, n) are
    the ENU direction cosines of the Sun and (Δe, Δn, Δu) is the baseline
    vector in metres from `config.json`.

    Cross-baseline amplitudes are normalised by `sqrt(|V_auto_r1| · |V_auto_r2|)`
    so the comparison is gain-independent. The simple per-antenna gain solution
    uses all cross-baselines to a chosen reference receiver via
    `G_r ≈ mean(V_obs[ref,r] / V_model[ref,r])` over time, averaged per channel.

    Launch with `marimo run scripts/sun_calibration.py`.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import h5py
    import json
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.coordinates import AltAz, EarthLocation, get_sun
    from astropy.time import Time
    import astropy.units as u

    C_LIGHT = 299792458.0  # m/s

    return AltAz, C_LIGHT, EarthLocation, Path, Time, get_sun, h5py, json, mo, np, plt, u


@app.cell
def _(Path, mo):
    dir_input = mo.ui.text(
        value=str(Path.cwd()),
        label="Directory",
        placeholder="Absolute path to search",
        full_width=True,
    )
    dir_input
    return (dir_input,)


@app.cell
def _(Path, dir_input, mo):
    _dir = Path(dir_input.value)
    _matches = (
        sorted(
            p for p in _dir.iterdir()
            if p.name.startswith("visibilities_") and p.suffix in (".h5", ".hdf5")
        )
        if _dir.is_dir()
        else []
    )
    file_picker = mo.ui.dropdown(
        options={p.name: p for p in _matches},
        value=_matches[0].name if _matches else None,
        label="Visibility file (visibilities_*.h5/.hdf5)",
        searchable=True,
    )
    file_picker
    return (file_picker,)


@app.cell
def _(Path, dir_input, mo):
    _candidates = [
        Path(dir_input.value) / "config.json",
        Path(dir_input.value).parent / "config.json",
        Path(__file__).parent.parent / "config.json",
    ]
    _default = next((str(p) for p in _candidates if p.exists()), "config.json")
    config_input = mo.ui.text(
        value=_default,
        label="config.json path",
        full_width=True,
    )
    config_input
    return (config_input,)


@app.cell
def _(Path, config_input, json, mo, np):
    _cfg_path = Path(config_input.value)
    mo.stop(
        not _cfg_path.exists(),
        mo.callout(f"`{_cfg_path}` not found — adjust the path above.", kind="danger"),
    )
    with open(_cfg_path) as _f:
        _cfg = json.load(_f)

    array_location = _cfg["array_location"]
    freq_plan = _cfg["frequency_plan"]

    # Flatten FPGA-keyed antenna_positions → {antenna_id: np.array([east, north, up])}
    antenna_enu = {}
    for _fpga_antennas in _cfg.get("antenna_positions", {}).values():
        for _ant_str, _pos in _fpga_antennas.items():
            antenna_enu[int(_ant_str)] = np.array([_pos["east"], _pos["north"], _pos["up"]])

    return antenna_enu, array_location, freq_plan


@app.cell
def _(file_picker, h5py, mo, np):
    mo.stop(
        file_picker.value is None,
        mo.callout("Choose a visibility HDF5 file to begin.", kind="info"),
    )
    visibility_path = file_picker.value
    with h5py.File(visibility_path, "r") as _hdf:
        mjd_start = float(_hdf.attrs.get("mjd_start", 0.0))
        min_channel = int(_hdf.attrs.get("min_channel", 0))
        vis_shape = tuple(_hdf["visibilities"].shape) if "visibilities" in _hdf else None
        antenna_ids = (
            _hdf["antenna_ids"][:].astype(int)
            if "antenna_ids" in _hdf
            else np.array([], dtype=int)
        )
        baseline_receivers = (
            _hdf["baseline_receiver_indices"][:].astype(int)
            if "baseline_receiver_indices" in _hdf
            else np.empty((0, 2), dtype=int)
        )
        baseline_antennas = (
            _hdf["baseline_antenna_ids"][:].astype(int)
            if "baseline_antenna_ids" in _hdf
            else np.empty((0, 2), dtype=int)
        )
        baseline_zeroed = (
            _hdf["baseline_zeroed"][:].astype(bool)
            if "baseline_zeroed" in _hdf
            else np.zeros(len(baseline_receivers) if baseline_receivers.size else 0, dtype=bool)
        )
        vis_raw = _hdf["visibilities"][:] if "visibilities" in _hdf else None

    if baseline_antennas.size == 0 and antenna_ids.size > 0 and baseline_receivers.size > 0:
        baseline_antennas = np.stack(
            [antenna_ids[baseline_receivers[:, 0]], antenna_ids[baseline_receivers[:, 1]]],
            axis=1,
        )

    return (
        antenna_ids,
        baseline_antennas,
        baseline_receivers,
        baseline_zeroed,
        min_channel,
        mjd_start,
        vis_raw,
        vis_shape,
        visibility_path,
    )


@app.cell
def _(np, vis_raw):
    # (n_int, n_chan, n_baseline, pol1, pol2) complex
    vis_complex = vis_raw[..., 0] + 1j * vis_raw[..., 1] if vis_raw is not None else None
    return (vis_complex,)


@app.cell
def _(mo, vis_shape):
    mo.stop(vis_shape is None, mo.callout("No `visibilities` dataset found.", kind="danger"))
    _n_int = vis_shape[0]
    integration_picker = mo.ui.slider(
        start=0, stop=_n_int - 1, value=0, label="Integration",
    )
    pol_picker = mo.ui.dropdown(
        options={"XX": (0, 0), "XY": (0, 1), "YX": (1, 0), "YY": (1, 1)},
        value="XX",
        label="Polarization",
    )
    time_override_toggle = mo.ui.checkbox(label="Override observation time", value=False)
    time_override_input = mo.ui.text(
        value="",
        placeholder="e.g. 2026-02-24T03:22:00 (UTC)",
        label="UTC time override",
    )
    mo.vstack([
        mo.hstack([integration_picker, pol_picker], justify="start"),
        mo.hstack([time_override_toggle, time_override_input], justify="start"),
    ])
    return integration_picker, pol_picker, time_override_input, time_override_toggle


@app.cell
def _(
    AltAz,
    EarthLocation,
    Time,
    array_location,
    get_sun,
    mjd_start,
    mo,
    np,
    time_override_input,
    time_override_toggle,
    u,
):
    if time_override_toggle.value and time_override_input.value.strip():
        _obs_time = Time(time_override_input.value.strip(), scale="utc")
    else:
        mo.stop(
            mjd_start == 0.0,
            mo.callout("No timestamp in file — enable the override above.", kind="warn"),
        )
        _obs_time = Time(mjd_start, format="mjd", scale="utc")

    _site = EarthLocation(
        lat=array_location["latitude_deg"] * u.deg,
        lon=array_location["longitude_deg"] * u.deg,
        height=array_location["height_m"] * u.m,
    )
    _sun = get_sun(_obs_time).transform_to(AltAz(obstime=_obs_time, location=_site))
    sun_alt_deg = float(_sun.alt.deg)
    sun_az_deg = float(_sun.az.deg)
    obs_time_str = _obs_time.iso

    # ENU direction cosines for the Sun's position
    _az_r, _alt_r = np.radians(sun_az_deg), np.radians(sun_alt_deg)
    sun_enu = np.array([
        np.sin(_az_r) * np.cos(_alt_r),  # East
        np.cos(_az_r) * np.cos(_alt_r),  # North
        np.sin(_alt_r),                   # Up
    ])

    return obs_time_str, sun_alt_deg, sun_az_deg, sun_enu


@app.cell(hide_code=True)
def _(mo, obs_time_str, sun_alt_deg, sun_az_deg, sun_enu):
    mo.vstack([
        mo.md(f"## Sun at {obs_time_str}"),
        mo.md(
            f"**Altitude:** {sun_alt_deg:.2f}°  &nbsp;&nbsp; "
            f"**Azimuth:** {sun_az_deg:.2f}°  \n"
            f"**ENU direction cosines:** l={sun_enu[0]:.4f} (E), "
            f"m={sun_enu[1]:.4f} (N), n={sun_enu[2]:.4f} (U)"
        ),
        mo.callout("Sun is below the horizon.", kind="warn")
        if sun_alt_deg < 0
        else mo.md(""),
    ])
    return


@app.cell
def _(
    C_LIGHT,
    antenna_enu,
    baseline_antennas,
    baseline_zeroed,
    freq_plan,
    min_channel,
    np,
    sun_enu,
    vis_shape,
):
    _n_chan = vis_shape[1]
    freqs_hz = (
        freq_plan["base_frequency_hz"]
        + (min_channel + np.arange(_n_chan)) * freq_plan["channel_bandwidth_hz"]
    )

    _n_bl = len(baseline_antennas)
    _ants1 = baseline_antennas[:, 0]
    _ants2 = baseline_antennas[:, 1]

    # ENU position for each antenna in each baseline slot (zero if unknown/disconnected)
    _pos1 = np.array([antenna_enu.get(a, np.zeros(3)) if a >= 0 else np.zeros(3) for a in _ants1])
    _pos2 = np.array([antenna_enu.get(a, np.zeros(3)) if a >= 0 else np.zeros(3) for a in _ants2])
    _known = (
        np.array([a >= 0 and antenna_enu.get(a) is not None for a in _ants1])
        & np.array([a >= 0 and antenna_enu.get(a) is not None for a in _ants2])
        & ~baseline_zeroed
    )

    _delta = _pos2 - _pos1                        # (n_bl, 3)
    _path = _delta @ sun_enu                       # (n_bl,)  geometric delay in metres
    _phase = 2 * np.pi / C_LIGHT * np.outer(_path, freqs_hz)  # (n_bl, n_chan)

    baseline_has_model = _known
    model_vis = np.where(_known[:, None], np.exp(1j * _phase), np.nan + 0j)

    return baseline_has_model, freqs_hz, model_vis


@app.cell
def _(baseline_receivers, np, vis_complex):
    # Autocorrelation indices: receiver pairs where r1 == r2
    _is_auto = baseline_receivers[:, 0] == baseline_receivers[:, 1]
    _auto_recv = baseline_receivers[_is_auto, 0]   # which receiver each autocorr belongs to
    _auto_idx = np.where(_is_auto)[0]              # baseline indices of autocorrelations

    # auto_amp[recv, chan, pol] = |V_auto| amplitude
    _n_recv_total = int(baseline_receivers.max()) + 1 if baseline_receivers.size else 0
    _n_int, _n_chan = vis_complex.shape[:2]
    auto_amp = np.zeros((_n_recv_total, _n_chan, 2))
    for _i, _bidx in zip(_auto_recv, _auto_idx):
        # Mean autocorrelation amplitude over integrations, XX and YY
        auto_amp[_i, :, 0] = np.abs(vis_complex[:, :, _bidx, 0, 0]).mean(axis=0)
        auto_amp[_i, :, 1] = np.abs(vis_complex[:, :, _bidx, 1, 1]).mean(axis=0)

    return auto_amp,


@app.cell
def _(
    baseline_antennas,
    baseline_has_model,
    baseline_receivers,
    mo,
    np,
):
    _is_cross = baseline_receivers[:, 0] != baseline_receivers[:, 1]
    _valid = baseline_has_model & _is_cross
    _options = {
        f"B{int(_b)}: A{int(baseline_antennas[_b, 0])}–A{int(baseline_antennas[_b, 1])} "
        f"(R{int(baseline_receivers[_b, 0])}–R{int(baseline_receivers[_b, 1])})"
        : int(_b)
        for _b in np.where(_valid)[0]
    }
    baseline_picker = mo.ui.dropdown(
        options=_options,
        value=next(iter(_options)) if _options else None,
        label="Baseline",
        searchable=True,
    )
    baseline_picker
    return (baseline_picker,)


@app.cell(hide_code=True)
def _(
    auto_amp,
    baseline_picker,
    baseline_receivers,
    freqs_hz,
    integration_picker,
    mo,
    model_vis,
    np,
    plt,
    pol_picker,
    vis_complex,
):
    mo.stop(baseline_picker.value is None, mo.callout("No valid baselines with model.", kind="warn"))
    _b = baseline_picker.value
    _t = integration_picker.value
    _p1, _p2 = pol_picker.value
    _r1, _r2 = baseline_receivers[_b, 0], baseline_receivers[_b, 1]

    _obs = vis_complex[_t, :, _b, _p1, _p2]           # (n_chan,) observed
    _mod = model_vis[_b]                               # (n_chan,) model (unit amplitude)
    _freqs_mhz = freqs_hz / 1e6

    # Normalise observed amplitude by geometric mean of autocorrelations
    _auto_r1 = auto_amp[_r1, :, _p1]
    _auto_r2 = auto_amp[_r2, :, _p2]
    _norm = np.sqrt(np.maximum(_auto_r1 * _auto_r2, 1e-30))
    _obs_norm_amp = np.abs(_obs) / _norm              # coherence magnitude (≈1 for perfect pt src)

    _obs_phase = np.degrees(np.angle(_obs))
    _mod_phase = np.degrees(np.angle(_mod))
    _residual = np.degrees(np.angle(_obs * np.conj(_mod)))  # wrap-safe phase residual

    _figure, _axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)

    # Amplitude
    _axes[0].plot(_freqs_mhz, _obs_norm_amp, label="Observed (normalised)", color="steelblue")
    _axes[0].axhline(1.0, color="tomato", linestyle="--", linewidth=1, label="Model (unit)")
    _axes[0].set_ylabel("Coherence amplitude")
    _axes[0].set_title(f"Amplitude — baseline {_b}, integration {_t}, pol {pol_picker.value}")
    _axes[0].legend()
    _axes[0].set_ylim(bottom=0)

    # Phase
    _axes[1].plot(_freqs_mhz, _obs_phase, label="Observed", color="steelblue")
    _axes[1].plot(_freqs_mhz, _mod_phase, label="Model (Sun)", color="tomato", linestyle="--")
    _axes[1].set_ylabel("Phase (°)")
    _axes[1].set_title("Phase")
    _axes[1].legend()
    _axes[1].set_ylim(-180, 180)

    # Residual
    _axes[2].plot(_freqs_mhz, _residual, color="darkorange")
    _axes[2].axhline(0, color="gray", linewidth=0.8)
    _axes[2].set_ylabel("Residual phase (°)")
    _axes[2].set_xlabel("Frequency (MHz)")
    _axes[2].set_title("Phase residual (observed − model)")
    _axes[2].set_ylim(-180, 180)

    mo.vstack([mo.md("## Per-baseline comparison"), _figure])
    return


@app.cell(hide_code=True)
def _(
    baseline_has_model,
    baseline_receivers,
    freqs_hz,
    mo,
    model_vis,
    np,
    plt,
    vis_complex,
    vis_shape,
):
    _n_int = vis_shape[0]
    _is_cross = baseline_receivers[:, 0] != baseline_receivers[:, 1]
    _valid_bl = np.where(baseline_has_model & _is_cross)[0]

    # Phase residual averaged over integrations for all valid cross-baselines
    # shape: (n_valid_bl, n_chan)
    _obs_mean = vis_complex[:, :, _valid_bl, 0, 0].mean(axis=0)  # mean over integrations, XX pol
    _mod = model_vis[_valid_bl]
    _residual_all = np.degrees(np.angle(_obs_mean * np.conj(_mod)))

    _figure, _axes = plt.subplots(1, 2, figsize=(14, max(4, len(_valid_bl) // 8)), constrained_layout=True)
    _freqs_mhz = freqs_hz / 1e6

    _im0 = _axes[0].imshow(
        np.abs(_obs_mean),
        aspect="auto", origin="upper", interpolation="nearest",
    )
    _axes[0].set_title("Observed amplitude (XX, mean over integrations)")
    _axes[0].set_xlabel("Channel index")
    _axes[0].set_ylabel("Baseline index (valid cross-baselines)")
    _figure.colorbar(_im0, ax=_axes[0], label="Amplitude")

    _im1 = _axes[1].imshow(
        _residual_all,
        aspect="auto", origin="upper", interpolation="nearest",
        vmin=-180, vmax=180, cmap="RdBu",
    )
    _axes[1].set_title("Phase residual: obs − model (XX, mean over integrations)")
    _axes[1].set_xlabel("Channel index")
    _axes[1].set_ylabel("Baseline index (valid cross-baselines)")
    _figure.colorbar(_im1, ax=_axes[1], label="Residual phase (°)")

    mo.vstack([mo.md("## All-baseline overview (XX, time-averaged)"), _figure])
    return


@app.cell(hide_code=True)
def _(
    antenna_ids,
    baseline_has_model,
    baseline_receivers,
    freqs_hz,
    mo,
    model_vis,
    np,
    plt,
    vis_complex,
    vis_shape,
):
    # Simple gain solution: for a chosen reference receiver (first non-disconnected
    # with valid model), solve G_r ≈ mean_t(V_obs[ref,r] / V_model[ref,r])
    _is_cross = baseline_receivers[:, 0] != baseline_receivers[:, 1]
    _valid_bl_mask = baseline_has_model & _is_cross

    # Pick reference receiver: one with the most valid baselines to others
    _recv_count = np.bincount(
        baseline_receivers[_valid_bl_mask].ravel(),
        minlength=int(baseline_receivers.max()) + 1 if baseline_receivers.size else 1,
    )
    _ref_recv = int(np.argmax(_recv_count))

    # Baselines involving the reference receiver (in either slot)
    _ref_bl_mask = _valid_bl_mask & (
        (baseline_receivers[:, 0] == _ref_recv) | (baseline_receivers[:, 1] == _ref_recv)
    )
    _ref_bls = np.where(_ref_bl_mask)[0]

    _n_int = vis_shape[0]
    _n_chan = vis_shape[1]
    _n_recv = int(baseline_receivers.max()) + 1 if baseline_receivers.size else 0
    gain_amp = np.full((_n_recv, _n_chan), np.nan)
    gain_phase_deg = np.full((_n_recv, _n_chan), np.nan)
    solved_receivers = []

    for _bidx in _ref_bls:
        _r0, _r1 = baseline_receivers[_bidx, 0], baseline_receivers[_bidx, 1]
        # Which one is the non-reference receiver?
        _other = _r1 if _r0 == _ref_recv else _r0
        _sign = 1 if _r0 == _ref_recv else -1  # V[ref,r] = G_ref * conj(G_r); V[r,ref] = G_r * conj(G_ref)

        _obs = vis_complex[:, :, _bidx, 0, 0]  # (n_int, n_chan) XX
        _mod = model_vis[_bidx][np.newaxis, :]  # (1, n_chan)
        _ratio = np.mean(_obs / _mod, axis=0)   # (n_chan,)

        # G_other = conj(_ratio) if ref is first slot, else _ratio
        _g = np.conj(_ratio) if _sign == 1 else _ratio
        gain_amp[_other] = np.abs(_g)
        gain_phase_deg[_other] = np.degrees(np.angle(_g))
        solved_receivers.append(_other)

    _solved = sorted(set(solved_receivers))
    _ant_labels = [
        f"R{r} (A{antenna_ids[r]})" if r < len(antenna_ids) else f"R{r}"
        for r in _solved
    ]
    _freqs_mhz = freqs_hz / 1e6

    _figure, _axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    for _r, _lbl in zip(_solved, _ant_labels):
        _axes[0].plot(_freqs_mhz, gain_amp[_r], label=_lbl, linewidth=0.8)
        _axes[1].plot(_freqs_mhz, gain_phase_deg[_r], label=_lbl, linewidth=0.8)

    _axes[0].set_ylabel("Gain amplitude")
    _axes[0].set_title(f"Per-antenna gain amplitude (reference: R{_ref_recv})")
    _axes[0].set_ylim(bottom=0)

    _axes[1].set_ylabel("Gain phase (°)")
    _axes[1].set_xlabel("Frequency (MHz)")
    _axes[1].set_title("Per-antenna gain phase")
    _axes[1].set_ylim(-180, 180)
    _axes[1].axhline(0, color="gray", linewidth=0.8)

    if len(_solved) <= 20:
        _axes[0].legend(fontsize=7, ncol=2)
        _axes[1].legend(fontsize=7, ncol=2)

    mo.vstack([mo.md(f"## Gain solutions (reference R{_ref_recv}, XX pol)"), _figure])
    return


if __name__ == "__main__":
    app.run()
